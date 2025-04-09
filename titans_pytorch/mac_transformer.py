from __future__ import annotations
from typing import Callable, Optional, Tuple, List

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat, Tensor # Added Tensor type hint
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# Import necessary components from the reference library
# Ensure titans-pytorch is installed or in the Python path
try:
    from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState # Assuming NeuralMemState is used/returned
    from titans_pytorch.memory_models import MemoryMLP
    from hyper_connections import get_init_and_expand_reduce_stream_functions
    from axial_positional_embedding import ContinuousAxialPositionalEmbedding
except ImportError:
    print("Please install titans-pytorch and its dependencies (axial_positional_embedding, hyper_connections)")
    # Define placeholders if import fails, code won't run
    Module = nn.Module
    NeuralMemory = None
    MemoryMLP = None
    get_init_and_expand_reduce_stream_functions = lambda *args, **kwargs: (None, lambda x: x, lambda x: x)
    ContinuousAxialPositionalEmbedding = lambda *args, **kwargs: nn.Identity()
    NeuralMemState = namedtuple('NeuralMemState', ['seq_index', 'weights', 'cache_store_segment', 'states', 'updates'])


# Import transformers library for LLM
try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer, GenerationConfig
except ImportError:
    print("Please install transformers library: pip install transformers")
    AutoModel, AutoConfig, AutoTokenizer, GenerationConfig = None, None, None, None

# einstein notation related
from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# constants
LinearNoBias = partial(Linear, bias = False)

# helpers (assuming these are available globally or defined elsewhere)
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
class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# --- Modified MAC Transformer ---

# Ensure necessary imports are present above this code:
# import torch, deepcopy, partial, etc.
# from torch import nn, Module, ModuleList, Linear, Parameter
# from titans_pytorch.neural_memory import NeuralMemory
# from titans_pytorch.memory_models import MemoryMLP
# --- Import the correct AutoClass ---
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer # Changed AutoModel
# --- End Import Change ---
# from hyper_connections import get_init_and_expand_reduce_stream_functions
# from axial_positional_embedding import ContinuousAxialPositionalEmbedding
# from einops.layers.torch import Rearrange
# from einops import repeat, rearrange, pack, unpack, einsum

# Helper functions (exists, default) assumed to be defined

class MemoryAsContextTransformerWithLLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        llm_model_name: str = "Qwen/Qwen1.5-0.5B", # Default to Qwen 0.5B
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
        ltm_default_depth: int = 2, # Default depth for MemoryMLP
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        neural_mem_weight_residual = False,
        token_emb: Module | None = None,
    ):
        super().__init__()

        # --- Basic Setup ---
        if not all([AutoModelForCausalLM, AutoConfig, AutoTokenizer]): # Check correct import
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
        # *** THE FIX IS HERE: Use AutoModelForCausalLM ***
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True)
        # *** END FIX ***
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)

        if llm_frozen:
            print("Freezing LLM weights.")
            # Note: Freezing behavior might differ slightly for *ForCausalLM models
            # Usually freezes the base model, but check if LM head needs separate handling if desired.
            for param in self.llm.parameters(): # Freezes entire model including head
                param.requires_grad = False
            self.llm.eval()

        # Get hidden dim from config (usually base model config is accessible)
        self.llm_hidden_dim = self.llm.config.hidden_size
        print(f"LLM hidden dim: {self.llm_hidden_dim}, Model dim: {dim}")
        # Projection layer remains the same
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
        # This projection might become redundant if using the LLM's LM head directly
        # Depending on how the forward pass is structured.
        # For now, keep it assuming the main forward pass still outputs hidden states.
        self.to_logits = LinearNoBias(dim, num_tokens)
        self.gate_llm_output = neural_mem_gate_attn_output
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    # ... (Keep the format_input_for_llm, forward, and sample methods as defined before) ...
    # Make sure the rest of the class definition follows here
    def format_input_for_llm(self, persistent_mem, current_segment_embeds):
        """
        Formats input for the LLM (Qwen1.5-0.5B).
        Assumes h_t (history) is NOT explicitly prepended here, simplifying the logic.
        Focuses on combining persistent memory (P) and the current segment (S_t).

        Args:
            persistent_mem (Tensor | None): Shape (num_persist, dim) or None.
            current_segment_embeds (Tensor): Shape (batch, seg_len, dim).

        Returns:
            inputs_embeds (Tensor): Shape (batch, combined_len, dim).
            attention_mask (Tensor): Shape (batch, combined_len).
            segment_len (int): Original length of current_segment_embeds.
        """
        batch_size, segment_len, dim = current_segment_embeds.shape
        device = current_segment_embeds.device
        inputs_list = []
        mask_list = []

        # 1. Persistent Memory
        len_p = 0
        if exists(persistent_mem):
            len_p = persistent_mem.shape[0]
            p_expanded = persistent_mem.unsqueeze(0).expand(batch_size, -1, -1) # (batch, num_persist, dim)
            inputs_list.append(p_expanded)
            mask_list.append(torch.ones(batch_size, len_p, dtype=torch.long, device=device))

        # 2. Current Segment
        inputs_list.append(current_segment_embeds)
        mask_list.append(torch.ones(batch_size, segment_len, dtype=torch.long, device=device))

        # 3. Combine
        combined_embeds = torch.cat(inputs_list, dim=1) # (batch, len_p + seg_len, dim)
        attention_mask = torch.cat(mask_list, dim=1) # (batch, len_p + seg_len)

        return combined_embeds, attention_mask, segment_len # Return original segment len too

    def forward(
        self,
        x, # Input token IDs: (batch, seq_len)
        return_loss = False, # Flag to compute loss
        ltm_state: Optional[NeuralMemState] = None, # Allow passing initial LTM state
        return_ltm_state: bool = False # Flag to return final LTM state
    ):

        if return_loss: # Prepare inputs and labels for loss calculation
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape # Get batch size and sequence length
        device = x.device

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
            # Assuming h_t is not explicitly passed here based on previous decision
            history_mem_embeds = None

            llm_input_embeds, llm_attention_mask, orig_segment_len = self.format_input_for_llm(
                persistent_mem_embeds,
                current_segment_embeds
            )

            # Call the LLM
            # The output of AutoModelForCausalLM is different from AutoModel
            # It usually returns CausalLMOutputWithPast object including logits and hidden_states
            context_manager = torch.no_grad() if not self.llm.training else torch.enable_grad()
            with context_manager:
                 llm_outputs = self.llm(
                     inputs_embeds=llm_input_embeds,
                     attention_mask=llm_attention_mask,
                     output_hidden_states=True, # Ensure hidden states are returned
                     return_dict=True
                 )
            # Use the last hidden state from the base model output if needed
            # The structure might be llm_outputs.hidden_states[-1] or similar
            # Check the specific output format for Qwen2ForCausalLM
            if hasattr(llm_outputs, 'hidden_states') and llm_outputs.hidden_states is not None:
                 y_t = llm_outputs.hidden_states[-1] # Get last hidden state
            else:
                 # Fallback or error handling if hidden states aren't available
                 # May need to access llm_outputs.logits and skip projection if using LM head directly
                 raise ValueError("Could not retrieve hidden states from LLM output.")


            # Extract the part corresponding to the original segment and project
            y_t_processed = y_t[:, -orig_segment_len:, :] # Take the last part corresponding to S_t
            y_t_projected = self.llm_output_proj(y_t_processed) # Project to model dim

            # Apply optional gating from LTM
            if exists(llm_output_gates):
                 # Ensure gates have compatible shape (batch, orig_segment_len, dim)
                 y_t_projected = y_t_projected * llm_output_gates[:, -orig_segment_len:, :] # Apply gating

            # Add residual connection for the LLM block
            x = add_residual_llm(y_t_projected)
            mem_input_layers.append(x) # Store LLM block output

            # --- FeedForward ---
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            x = add_ff_residual(ff_out)
            mem_input_layers.append(x)

        # --- Final Output ---
        x = self.reduce_streams(x) # Reduce streams if hyper-connections used
        x = self.norm(x) # Final normalization
        logits = self.to_logits(x) # Project to vocabulary logits

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
        seq_len: int, # Total desired sequence length (including prompt)
        temperature=1.0,
        filter_thres=0.9, # Example filter (top_p)
        **kwargs # Allow passing other generation config args
    ):
        """
        Basic sampling using the LLM's .generate() method.
        WARNING: This method DOES NOT update the LTM state during generation.
        A custom generation loop is needed for stateful LTM sampling.
        """
        self.eval() # Ensure model is in eval mode

        if not exists(self.tokenizer):
             print("Warning: Tokenizer not loaded. Cannot use LLM generate. Returning prompt.")
             return prompt
        if not hasattr(self.llm, 'generate'):
             print("Warning: Loaded LLM does not have a .generate() method. Cannot sample.")
             return prompt

        # Ensure prompt is on the same device as the LLM
        device = next(self.llm.parameters()).device
        prompt = prompt.to(device)

        print("Warning: Using basic LLM generate, LTM state is NOT updated during sampling.")

        # Prepare generation config
        gen_config = GenerationConfig(
             max_length=seq_len,
             temperature=temperature,
             top_p=filter_thres,
             pad_token_id=self.tokenizer.eos_token_id, # Use EOS for padding
             eos_token_id=self.tokenizer.eos_token_id,
             **kwargs
        )

        # Generate sequence using the underlying LLM directly
        # Now self.llm is AutoModelForCausalLM, so it has .generate()
        output_ids = self.llm.generate(
             input_ids=prompt,
             generation_config=gen_config
        )

        # Return only the generated part (excluding the prompt)
        prompt_len = prompt.shape[1]
        return output_ids[:, prompt_len:]

    # ... (Keep the format_input_for_llm, forward, and sample methods as defined before) ...
    # Make sure the rest of the class definition follows here
    def format_input_for_llm(self, persistent_mem, current_segment_embeds):
        """
        Formats input for the LLM (Qwen1.5-0.5B).
        Assumes h_t (history) is NOT explicitly prepended here, simplifying the logic.
        Focuses on combining persistent memory (P) and the current segment (S_t).

        Args:
            persistent_mem (Tensor | None): Shape (num_persist, dim) or None.
            current_segment_embeds (Tensor): Shape (batch, seg_len, dim).

        Returns:
            inputs_embeds (Tensor): Shape (batch, combined_len, dim).
            attention_mask (Tensor): Shape (batch, combined_len).
            segment_len (int): Original length of current_segment_embeds.
        """
        batch_size, segment_len, dim = current_segment_embeds.shape
        device = current_segment_embeds.device
        inputs_list = []
        mask_list = []

        # 1. Persistent Memory
        len_p = 0
        if exists(persistent_mem):
            len_p = persistent_mem.shape[0]
            p_expanded = persistent_mem.unsqueeze(0).expand(batch_size, -1, -1) # (batch, num_persist, dim)
            inputs_list.append(p_expanded)
            mask_list.append(torch.ones(batch_size, len_p, dtype=torch.long, device=device))

        # 2. Current Segment
        inputs_list.append(current_segment_embeds)
        mask_list.append(torch.ones(batch_size, segment_len, dtype=torch.long, device=device))

        # 3. Combine
        combined_embeds = torch.cat(inputs_list, dim=1) # (batch, len_p + seg_len, dim)
        attention_mask = torch.cat(mask_list, dim=1) # (batch, len_p + seg_len)

        # Note: Using a simple 'ones' mask. A causal mask might be needed depending on LLM and task.
        # combined_len = combined_embeds.shape[1]
        # causal_mask = torch.tril(torch.ones((combined_len, combined_len), dtype=torch.bool, device=device))
        # attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1) # Check LLM expected mask format

        return combined_embeds, attention_mask, segment_len # Return original segment len too

    def forward(
        self,
        x, # Input token IDs: (batch, seq_len)
        return_loss = False, # Flag to compute loss
        ltm_state: Optional[NeuralMemState] = None, # Allow passing initial LTM state
        return_ltm_state: bool = False # Flag to return final LTM state
    ):

        if return_loss: # Prepare inputs and labels for loss calculation
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape # Get batch size and sequence length
        device = x.device

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
            # Assuming h_t is not explicitly passed here based on previous decision
            history_mem_embeds = None

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
                     return_dict=True
                 )
            y_t = llm_outputs.last_hidden_state # (batch, combined_len, llm_hidden_dim)

            # Extract the part corresponding to the original segment and project
            y_t_processed = y_t[:, -orig_segment_len:, :] # Take the last part corresponding to S_t
            y_t_projected = self.llm_output_proj(y_t_processed) # Project to model dim

            # Apply optional gating from LTM
            if exists(llm_output_gates):
                 # Ensure gates have compatible shape (batch, orig_segment_len, dim)
                 y_t_projected = y_t_projected * llm_output_gates[:, -orig_segment_len:, :] # Apply gating

            # Add residual connection for the LLM block
            x = add_residual_llm(y_t_projected)
            mem_input_layers.append(x) # Store LLM block output

            # --- FeedForward ---
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            x = add_ff_residual(ff_out)
            mem_input_layers.append(x)

        # --- Final Output ---
        x = self.reduce_streams(x) # Reduce streams if hyper-connections used
        x = self.norm(x) # Final normalization
        logits = self.to_logits(x) # Project to vocabulary logits

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
        seq_len: int, # Total desired sequence length (including prompt)
        temperature=1.0,
        filter_thres=0.9, # Example filter (top_p)
        **kwargs # Allow passing other generation config args
    ):
        """
        Basic sampling using the LLM's .generate() method.
        WARNING: This method DOES NOT update the LTM state during generation.
        A custom generation loop is needed for stateful LTM sampling.
        """
        self.eval() # Ensure model is in eval mode

        if not exists(self.tokenizer):
             print("Warning: Tokenizer not loaded. Cannot use LLM generate. Returning prompt.")
             return prompt

        # Ensure prompt is on the same device as the LLM
        device = next(self.llm.parameters()).device
        prompt = prompt.to(device)

        print("Warning: Using basic LLM generate, LTM state is NOT updated during sampling.")

        # Prepare generation config
        gen_config = GenerationConfig(
             max_length=seq_len,
             temperature=temperature,
             top_p=filter_thres,
             pad_token_id=self.tokenizer.eos_token_id, # Use EOS for padding
             eos_token_id=self.tokenizer.eos_token_id,
             **kwargs
        )

        # Generate sequence using the underlying LLM directly
        output_ids = self.llm.generate(
             input_ids=prompt,
             generation_config=gen_config
        )

        # Return only the generated part (excluding the prompt)
        prompt_len = prompt.shape[1]
        return output_ids[:, prompt_len:]

    def format_input_for_llm(self, persistent_mem, current_segment_embeds):
        """
        Formats input for the LLM (Qwen1.5-0.5B).
        Assumes h_t (history) is NOT explicitly prepended here, simplifying the logic.
        Focuses on combining persistent memory (P) and the current segment (S_t).

        Args:
            persistent_mem (Tensor | None): Shape (num_persist, dim) or None.
            current_segment_embeds (Tensor): Shape (batch, seg_len, dim).

        Returns:
            inputs_embeds (Tensor): Shape (batch, combined_len, dim).
            attention_mask (Tensor): Shape (batch, combined_len).
            segment_len (int): Original length of current_segment_embeds.
        """
        batch_size, segment_len, dim = current_segment_embeds.shape
        device = current_segment_embeds.device
        inputs_list = []
        mask_list = []

        # 1. Persistent Memory
        len_p = 0
        if exists(persistent_mem):
            len_p = persistent_mem.shape[0]
            p_expanded = persistent_mem.unsqueeze(0).expand(batch_size, -1, -1) # (batch, num_persist, dim)
            inputs_list.append(p_expanded)
            mask_list.append(torch.ones(batch_size, len_p, dtype=torch.long, device=device))

        # 2. Current Segment
        inputs_list.append(current_segment_embeds)
        mask_list.append(torch.ones(batch_size, segment_len, dtype=torch.long, device=device))

        # 3. Combine
        combined_embeds = torch.cat(inputs_list, dim=1) # (batch, len_p + seg_len, dim)
        attention_mask = torch.cat(mask_list, dim=1) # (batch, len_p + seg_len)

        # Note: A simple 'ones' mask allows full attention. More complex causal masking
        # or preventing attention between P and S_t might be needed depending on the goal.
        # Qwen generally uses a standard causal mask structure if positional embeddings are implicit.
        # If using inputs_embeds, providing a causal mask is often recommended.
        # Let's create a basic causal mask for the combined sequence.
        combined_len = combined_embeds.shape[1]
        # Standard lower-triangular causal mask
        # causal_mask = torch.tril(torch.ones((combined_len, combined_len), dtype=torch.bool, device=device))
        # Expand for batch: (batch, 1, combined_len, combined_len) -> Check Qwen expected format
        # Qwen models typically expect attention_mask shape (batch_size, sequence_length)
        # where 1 indicates attend and 0 indicates ignore.
        # The above 'ones' mask allows full attention. Let's stick with that for simplicity,
        # assuming Qwen's internal layers apply causality if needed.

        return combined_embeds, attention_mask, segment_len # Return original segment len too

    def forward(
        self,
        x, # Input token IDs: (batch, seq_len)
        return_loss = False,
        ltm_state: Optional[NeuralMemState] = None, # Allow passing initial LTM state
        return_ltm_state: bool = False # Flag to return final LTM state
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape
        device = x.device

        # --- Initial Processing ---
        x_embed = self.token_emb(x) # (batch, seq_len, dim)
        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len, (self.neural_memory_segment_len,))
        x_embed = x_embed + pos_emb.to(device) # Ensure pos_emb is on correct device

        # --- LTM State ---
        current_ltm_state = ltm_state # Use passed state or None (NeuralMemory handles init)
        mem_weight_residual = None

        # --- Layer Loop ---
        x_residual = x_embed # Input to the first layer
        x = self.expand_streams(x_residual) # Expand for hyper-connections
        mem_input_layers = [] # Store intermediate layer outputs

        for layer_idx, (mem_hyper_conn, llm_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, _, ff) in enumerate(self.layers):

            layer_input = x # Store input to this layer block for potential LTM QKV selection
            mem_input_layers.append(layer_input)

            ltm_output = None # Output from LTM module
            llm_output_gates = None # Gating signal from LTM

            # --- Neural Memory (LTM) ---
            if exists(mem):
                mem_input, add_residual_mem = mem_hyper_conn(x)
                qkv_mem_input = stack((mem_input, mem_input, mem_input)) # Default: use same input for QKV
                if exists(mem_qkv_layer_selector):
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                # Call LTM forward - Assuming it handles updates and returns output + state
                # The exact signature/return depends on NeuralMemory implementation
                # We need to pass the *embeddings* `x_embed` or derived features here, not token IDs `x`
                # Let's assume qkv_mem_input is derived correctly from features
                ltm_output, current_ltm_state = mem.forward(
                    qkv_mem_input, # Should be features, not IDs
                    state = current_ltm_state,
                    prev_weights = mem_weight_residual
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
            llm_input_stream, add_residual_llm = llm_hyper_conn(x)
            mem_input_layers.append(llm_input_stream) # Store LLM block input

            # Prepare input for LLM
            # Uses persistent memory and the current input stream to the LLM block
            # Assumes h_t is implicitly handled by LTM state/updates, not prepended
            current_segment_embeds = llm_input_stream
            persistent_mem_embeds = self.persist_mems if exists(self.persist_mems) else None

            llm_input_embeds, llm_attention_mask, orig_segment_len = self.format_input_for_llm(
                persistent_mem_embeds,
                current_segment_embeds
            )

            # Call the LLM
            # Use torch.no_grad() if LLM is frozen to save memory
            context_manager = torch.no_grad() if self.llm.training == False else torch.enable_grad() # Check if llm is frozen
            with context_manager:
                 llm_outputs = self.llm(
                     inputs_embeds=llm_input_embeds,
                     attention_mask=llm_attention_mask,
                     return_dict=True
                 )
            y_t = llm_outputs.last_hidden_state # (batch, combined_len, llm_hidden_dim)

            # Extract the part corresponding to the original segment and project
            # Output shape needs to be (batch, orig_segment_len, dim)
            y_t_processed = y_t[:, -orig_segment_len:, :] # Take the last part corresponding to S_t
            y_t_projected = self.llm_output_proj(y_t_processed) # Project to model dim

            # Apply optional gating from LTM
            if exists(llm_output_gates):
                 # Ensure gates have compatible shape (batch, orig_segment_len, dim)
                 # This might require adjusting how gating signal is derived/shaped
                 y_t_projected = y_t_projected * llm_output_gates[:, -orig_segment_len:, :] # Apply gating

            # Add residual connection for the LLM block
            x = add_residual_llm(y_t_projected)
            mem_input_layers.append(x) # Store LLM block output

            # --- FeedForward ---
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            x = add_ff_residual(ff_out)
            mem_input_layers.append(x)

        # --- Final Output ---
        x = self.reduce_streams(x) # Reduce streams if hyper-connections used
        x = self.norm(x) # Final normalization
        logits = self.to_logits(x) # Project to vocabulary logits

        # Prepare results
        result = (logits,)
        if return_loss:
            loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
            result = (loss,) + result
        if return_ltm_state:
            result = result + (current_ltm_state,)

        if len(result) == 1:
            return result[0] # Return only logits/loss if that's all requested

        return result # Return tuple (loss, logits, state) or (logits, state) or (logits,)


    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor, # Input token IDs: (batch, prompt_len)
        seq_len: int, # Total desired sequence length (including prompt)
        temperature=1.0,
        filter_thres=0.9, # Example filter (top_p)
        **kwargs # Allow passing other generation config args
    ):
        """
        Basic sampling using the LLM's .generate() method.
        WARNING: This method DOES NOT update the LTM state during generation.
        A custom generation loop is needed for stateful LTM sampling.
        """
        self.eval() # Ensure model is in eval mode

        if not exists(self.tokenizer):
             print("Warning: Tokenizer not loaded. Cannot use LLM generate. Returning prompt.")
             return prompt

        # Ensure prompt is on the same device as the LLM
        device = next(self.llm.parameters()).device
        prompt = prompt.to(device)

        # Use LLM's generate method
        # Note: This bypasses the custom layer structure and LTM updates.
        # It's only using the raw LLM for generation based on the prompt.
        # Input formatting for prompt might be needed if P is used during training.
        print("Warning: Using basic LLM generate, LTM state is NOT updated during sampling.")

        # Prepare generation config
        gen_config = GenerationConfig(
             max_length=seq_len,
             temperature=temperature,
             top_p=filter_thres,
             pad_token_id=self.tokenizer.eos_token_id, # Use EOS for padding
             eos_token_id=self.tokenizer.eos_token_id,
             **kwargs
        )

        # Generate sequence
        output_ids = self.llm.generate(
             input_ids=prompt,
             generation_config=gen_config
        )

        # Return only the generated part (excluding the prompt)
        prompt_len = prompt.shape[1]
        return output_ids[:, prompt_len:]

