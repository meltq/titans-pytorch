## Ensure necessary imports are present above this code:
import torch
from torch import nn, stack, cat, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter
from typing import Callable, Optional, Tuple, List
from copy import deepcopy
from functools import partial
from collections import namedtuple
import tqdm # Keep if sample method uses it internally

# Import necessary components from the reference library
try:
    from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState
    from titans_pytorch.memory_models import MemoryMLP
    from hyper_connections import get_init_and_expand_reduce_stream_functions
    from axial_positional_embedding import ContinuousAxialPositionalEmbedding
except ImportError:
    print("Please install titans-pytorch and its dependencies")
    # Define placeholders if import fails, code won't run
    Module = nn.Module
    NeuralMemory = None
    MemoryMLP = None
    get_init_and_expand_reduce_stream_functions = lambda *args, **kwargs: (None, lambda x: x, lambda x: x)
    ContinuousAxialPositionalEmbedding = lambda *args, **kwargs: nn.Identity()
    NeuralMemState = namedtuple('NeuralMemState', ['seq_index', 'weights', 'cache_store_segment', 'states', 'updates'])


# Import transformers library for LLM
try:
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, GenerationConfig
except ImportError:
    print("Please install transformers library: pip install transformers")
    AutoModelForCausalLM, AutoConfig, AutoTokenizer, GenerationConfig = None, None, None, None

# einstein notation related
from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# constants
LinearNoBias = partial(Linear, bias = False)

def exists(v):
    return v is not None
def default(v, d):
    return v if exists(v) else d
def identity(t):
    return t
# ... other helpers like divisible_by, round_up_multiple, pad_at_dim etc. ...
# Need pad_at_dim for format_input_for_llm mask creation if using complex masks
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# sampling related (assuming available)
def gumbel_sample(t, temperature = 1.):
     if temperature > 0.:
         noise = torch.rand_like(t)
         # Safe log
         eps = torch.finfo(t.dtype).eps
         noise = -torch.log(-torch.log(noise.clamp(min=eps, max=1-eps)).clamp(min=eps))
         t = t / temperature + noise
     return t.argmax(dim = -1, keepdim = True)
# ... other sampling helpers ...


# feedforward (assuming available)
class GEGLU(Module): # ... (Implementation) ...
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1); return F.silu(gate) * x
def FeedForward(dim, mult = 4): # ... (Implementation) ...
    dim_inner = int(dim * mult * 2 / 3); return nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim))


# --- Modified MAC Transformer ---

class MemoryAsContextTransformerWithLLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        llm_model_name: str = "Qwen/Qwen1.5-0.5B",
        llm_frozen: bool = True,
        segment_len,
        neural_memory_segment_len = None,
        neural_mem_gate_attn_output = False,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        neural_memory_batch_size = None,
        neural_memory_qkv_receives_diff_views = False,
        ff_mult = 4,
        num_residual_streams = 1,
        neural_memory_model: Module | None = None,
        ltm_default_depth: int = 2,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        neural_mem_weight_residual = False,
        token_emb: Module | None = None,
    ):
        super().__init__()

        # --- Basic Setup ---
        if not all([AutoModelForCausalLM, AutoConfig, AutoTokenizer]):
             raise ImportError("Please install the `transformers` library.")
        if not all([NeuralMemory, MemoryMLP]):
             raise ImportError("Could not import NeuralMemory/MemoryMLP from titans-pytorch.")

        if not exists(token_emb):
            self.token_emb = nn.Embedding(num_tokens, dim)
        else:
            self.token_emb = token_emb
        self.num_tokens = num_tokens
        self.model_dim = dim

        # --- Positional Embedding ---
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)

        # --- Persistent Memory (P) ---
        self.num_persist_mem_tokens = num_persist_mem_tokens
        if num_persist_mem_tokens > 0:
             self.persist_mems = nn.Parameter(torch.randn(num_persist_mem_tokens, dim) * 0.02)
        else:
             self.persist_mems = None

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.segment_len = segment_len

        # --- LLM Integration ---
        print(f"Loading LLM for Causal LM: {llm_model_name}")
        self.llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)

        if llm_frozen:
            print("Freezing LLM weights.")
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        self.llm_hidden_dim = self.llm.config.hidden_size
        print(f"LLM hidden dim: {self.llm_hidden_dim}, Model dim: {dim}")
        self.llm_output_proj = nn.Linear(self.llm_hidden_dim, dim) if self.llm_hidden_dim != dim else nn.Identity()

        # --- Hyper Connections ---
        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        # --- Layer Definition ---
        self.layers = ModuleList([])
        self.neural_memory_segment_len = default(neural_memory_segment_len, segment_len)
        layers_indices = tuple(range(1, depth + 1))
        neural_memory_layers = default(neural_memory_layers, layers_indices)
        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True

        for layer in layers_indices:
            is_first = layer == 1
            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)
                if not is_first and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1
                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )
                if exists(neural_memory_model):
                    ltm_model_instance = deepcopy(neural_memory_model)
                else:
                    print(f"Layer {layer}: Initializing default MemoryMLP(dim={dim}, depth={ltm_default_depth}) for LTM.")
                    ltm_model_instance = MemoryMLP(dim=dim, depth=ltm_default_depth)

                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = ltm_model_instance,
                    qkv_receives_diff_views = neural_memory_qkv_receives_diff_views,
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )
                is_first_neural_mem = False

            llm_hyper_conn = init_hyper_conn()
            ff_hyper_conn = init_hyper_conn()
            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                mem_hyper_conn, llm_hyper_conn, ff_hyper_conn,
                mem_qkv_layer_selector, mem, None, ff
            ]))

        # --- Final Output Layers ---
        self.norm = nn.RMSNorm(dim)
        self.to_logits = LinearNoBias(dim, num_tokens)
        self.gate_llm_output = neural_mem_gate_attn_output
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    # *** ADDED .device PROPERTY ***
    @property
    def device(self):
        # Return the device of the first parameter found (or a specific buffer/parameter)
        # This makes the model compatible with code expecting a .device attribute
        try:
            # A reliable parameter to check is usually the token embedding weight
            return self.token_emb.weight.device
        except AttributeError:
            # Fallback if token_emb isn't standard or parameters() is empty
            try:
                return next(self.parameters()).device
            except StopIteration:
                # If model has no parameters, maybe return device of a buffer
                try:
                    return next(self.buffers()).device
                except StopIteration:
                    # Default to CPU if no parameters or buffers found (unlikely for this model)
                    print("Warning: Could not determine model device automatically. Defaulting to CPU.")
                    return torch.device("cpu")

    def format_input_for_llm(self, persistent_mem, current_segment_embeds):
        # ... (Implementation from previous step) ...
        batch_size, segment_len, dim = current_segment_embeds.shape
        device = current_segment_embeds.device
        inputs_list = []
        mask_list = []
        len_p = 0
        if exists(persistent_mem):
            len_p = persistent_mem.shape[0]
            p_expanded = persistent_mem.unsqueeze(0).expand(batch_size, -1, -1)
            inputs_list.append(p_expanded)
            mask_list.append(torch.ones(batch_size, len_p, dtype=torch.long, device=device))
        inputs_list.append(current_segment_embeds)
        mask_list.append(torch.ones(batch_size, segment_len, dtype=torch.long, device=device))
        combined_embeds = torch.cat(inputs_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)
        return combined_embeds, attention_mask, segment_len


    def forward(
        self,
        x, # Input token IDs: (batch, seq_len)
        return_loss = False, # Flag to compute loss
        ltm_state: Optional[NeuralMemState] = None, # Allow passing initial LTM state
        return_ltm_state: bool = False # Flag to return final LTM state
    ):
        # ... (Implementation from previous step, ensure imports/helpers are defined) ...

        if return_loss: # Prepare inputs and labels for loss calculation
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape # Get batch size and sequence length
        device = x.device # Use the model's device property

        # --- Initial Processing ---
        x_embed = self.token_emb(x) # Convert token IDs to embeddings: (batch, seq_len, dim)
        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len, (self.neural_memory_segment_len,))
        x_embed = x_embed + pos_emb.to(device) # Add positional embeddings

        # --- LTM State ---
        current_ltm_state = ltm_state # Use passed state or None (NeuralMemory handles init)
        mem_weight_residual = None # For LTM weight residual connection

        # --- Layer Loop ---
        x_residual = x_embed # Input to the first layer
        x = self.expand_streams(x_residual) # Expand for hyper-connections
        mem_input_layers = [] # Store intermediate layer outputs

        for layer_idx, (mem_hyper_conn, llm_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, _, ff) in enumerate(self.layers): # Iterate through layer components

            layer_input = x # Store input to this layer block for potential LTM QKV selection
            mem_input_layers.append(layer_input)

            ltm_output = None # Output from LTM module
            llm_output_gates = None # Gating signal from LTM

            # --- Neural Memory (LTM) ---
            if exists(mem): # If this layer includes an LTM module
                mem_input, add_residual_mem = mem_hyper_conn(x) # Get input via hyper-conn
                qkv_mem_input = stack((mem_input, mem_input, mem_input)) # Default: use same input for QKV
                if exists(mem_qkv_layer_selector):
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                # Call LTM forward - Assuming it handles updates and returns output + state
                ltm_output, current_ltm_state = mem.forward(
                    qkv_mem_input, # Should be features, not IDs
                    state = current_ltm_state,
                    prev_weights = mem_weight_residual
                    # Pass other args like store_mask if needed
                )
                # Update weight residual if applicable
                if self.neural_mem_weight_residual and hasattr(current_ltm_state, 'updates'):
                    mem_weight_residual = current_ltm_state.updates

                # Apply LTM output (either gating or residual)
                if self.gate_llm_output:
                    llm_output_gates = ltm_output.sigmoid()
                else:
                    x = add_residual_mem(ltm_output)

            # Store LTM block output
            mem_input_layers.append(x)

            # --- LLM (Short-Term Memory) ---
            llm_input_stream, add_residual_llm = llm_hyper_conn(x) # Get input via hyper-conn
            mem_input_layers.append(llm_input_stream) # Store LLM block input

            # Prepare input for LLM
            current_segment_embeds = llm_input_stream
            persistent_mem_embeds = self.persist_mems if exists(self.persist_mems) else None
            history_mem_embeds = None # Assuming h_t is not explicitly passed

            llm_input_embeds, llm_attention_mask, orig_segment_len = self.format_input_for_llm(
                persistent_mem_embeds,
                current_segment_embeds
            )

            # Call the LLM
            context_manager = torch.no_grad() if not self.llm.training else torch.enable_grad()
            with context_manager:
                 llm_outputs = self.llm(
                     inputs_embeds=llm_input_embeds,
                     attention_mask=llm_attention_mask,
                     output_hidden_states=True,
                     return_dict=True
                 )
            if hasattr(llm_outputs, 'hidden_states') and llm_outputs.hidden_states is not None:
                 y_t = llm_outputs.hidden_states[-1]
            else:
                 raise ValueError("Could not retrieve hidden states from LLM output.")

            # Extract the part corresponding to the original segment and project
            y_t_processed = y_t[:, -orig_segment_len:, :]
            y_t_projected = self.llm_output_proj(y_t_processed)

            # Apply optional gating from LTM
            if exists(llm_output_gates):
                 y_t_projected = y_t_projected * llm_output_gates[:, -orig_segment_len:, :]

            # Add residual connection for the LLM block
            x = add_residual_llm(y_t_projected)
            mem_input_layers.append(x)

            # --- FeedForward ---
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            x = add_ff_residual(ff_out)
            mem_input_layers.append(x)

        # --- Final Output ---
        x = self.reduce_streams(x)
        x = self.norm(x)
        logits = self.to_logits(x)

        # Prepare results
        result = (logits,)
        if return_loss:
            loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
            result = (loss,) + result
        if return_ltm_state:
            result = result + (current_ltm_state,)

        if len(result) == 1:
            return result[0]
        return result


    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor, # Input token IDs: (batch, prompt_len)
        attention_mask: Optional[Tensor] = None, # Accept attention_mask
        seq_len: int = 2048,
        temperature=1.0,
        filter_thres=0.9, # Represents top_p
        do_sample: Optional[bool] = None,
        **kwargs
    ):
        # ... (Implementation from previous step - handles do_sample and passes mask) ...
        self.eval()
        if not exists(self.tokenizer): print("Warning: Tokenizer not loaded..."); return prompt
        if not hasattr(self.llm, 'generate'): print("Warning: LLM has no .generate()"); return prompt
        device = self.device # Use the new property
        prompt = prompt.to(device)
        if exists(attention_mask): attention_mask = attention_mask.to(device)
        print("Warning: Using basic LLM generate, LTM state is NOT updated during sampling.")
        model_gen_config = getattr(self.llm, "generation_config", None)
        gen_config = GenerationConfig.from_dict(model_gen_config.to_dict()) if model_gen_config else GenerationConfig()
        should_sample = (temperature != 1.0 or filter_thres < 1.0)
        gen_config.do_sample = default(do_sample, should_sample)
        gen_config.max_length = seq_len
        gen_config.temperature = temperature
        gen_config.top_p = filter_thres
        gen_config.pad_token_id = self.tokenizer.eos_token_id
        gen_config.eos_token_id = self.tokenizer.eos_token_id
        explicit_args = {'max_length', 'temperature', 'top_p', 'pad_token_id', 'eos_token_id', 'do_sample'}
        filtered_gen_kwargs = {}
        for key, value in kwargs.items():
            if key not in explicit_args and hasattr(gen_config, key): setattr(gen_config, key, value)
            elif key not in explicit_args: filtered_gen_kwargs[key] = value
        output_ids = self.llm.generate(
             input_ids=prompt,
             attention_mask=attention_mask, # Pass mask
             generation_config=gen_config,
             **filtered_gen_kwargs
        )
        prompt_len = prompt.shape[1]
        if output_ids.shape[1] > prompt_len: return output_ids[:, prompt_len:]
        else: return torch.tensor([], dtype=torch.long, device=device)
