# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just rebuilding
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY connection from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5 Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with chechable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Code Implementation Protocol
1. **ALWAYS Explain Before Implementing**:
   - Before writing ANY new file or significant code block, EXPLAIN the design first
   - Include: architecture, key methods, data structures, algorithms, tradeoffs
   - Wait for user confirmation before proceeding
   - This applies to: new files, new classes, complex functions, algorithms
2. **Explain Edits Before Making Them**:
   - Before editing existing code, describe what changes will be made and why
   - Show before/after snippets for complex changes
   - For simple edits (typos, formatting), can proceed directly
3. **Collaborative Implementation**:
   - User may want to make edits together in IDE
   - Default to explanation-first workflow unless explicitly told to just implement

## Core Principles
- **Simplicity**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Project Overview

Open Dreams is a JAX/Flax NNX research codebase for training world models—neural networks that learn to predict how environments evolve over time based on actions. The codebase implements diffusion-based world models using flow matching for stable, high-quality video prediction.

**Key Technologies:**
- **JAX/Flax NNX**: Core ML framework with automatic differentiation and JIT compilation
- **Hydra**: Configuration management with composable YAML configs
- **TensorFlow**: Data loading pipeline (TFRecords)
- **Orbax**: Checkpoint management
- **Weights & Biases**: Experiment tracking

## Code Quality Standards

### Logging vs Print Statements

**Rule:** Never use `print()` in model code or library functions. Only use in user-facing scripts.

```python
# ✅ GOOD - Model code uses logging
import logging
logger = logging.getLogger(__name__)

class LatentDiffusionWorldModel(BaseWorldModel):
    def _load_finetuned_vae(self, config, rngs):
        logger.info(f"Loading VAE from {config.vae_checkpoint_path}")
        # ... load logic
        logger.info(f"VAE loaded successfully from step {step}")

# ✅ GOOD - CLI scripts can use print for user output
def main(config):
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

# ❌ BAD - Model code using print
class WorldModel:
    def inference(self, inputs):
        print(f"Running inference with shape {inputs.shape}")  # NO!
```

**Debug prints:** Remove ALL debug print statements before committing. Use logging with `logger.debug()` if needed for development.

### Function Length Guidelines

**Philosophy:** Functions can start long while prototyping, but should be refactored to <100 lines during review passes.

**The "One Screen" Heuristic:** A function should fit on one screen so developers (and Claude!) can see the entire function at once without scrolling. This dramatically improves understandability and enables both humans and LLMs to reason about the complete logic in one context window.

**Examples:**

```python
# ❌ Needs refactoring - 154-line main function
def main(config):
    # 154 lines of initialization, loading, setup...
    # Too much to reason about at once

# ✅ GOOD - Refactored into focused functions
def main(config):
    devices = setup_devices(config)
    dataloader = create_dataloader(config, devices)
    model = load_model(config, devices)
    run_training(model, dataloader, config)

def setup_devices(config):
    # Device initialization logic (20 lines)
    ...

def create_dataloader(config, devices):
    # Dataloader setup (30 lines)
    ...
```

**For long `__init__` methods:** Extract helper methods:
```python
def __init__(self, config, rngs):
    super().__init__()
    self._init_config(config)
    self._init_vae(config, rngs)
    self._init_encoder_decoder(config, rngs)
    self._init_embeddings(config, rngs)
```

### Type Annotations

**Required:** All public functions must have complete type annotations.

```python
# ✅ GOOD
def encode_latents(self, images: Array) -> Array:
    """Encode images to latents."""
    ...

# ✅ GOOD - Explicit Any for dynamic types
def log_eval_metrics(
    self, eval_metrics: dict[str, Array], config: DictConfig
) -> dict[str, Any]:
    ...

# ❌ BAD - Missing return type
def log_device_memory(prefix: str):
    ...

# ✅ GOOD - Add return type
def log_device_memory(prefix: str) -> None:
    ...
```

### Style Compliance

**Reference:** [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

**Key rules:**
- **Line length:** 80 characters (Black formatter may extend to 88, but aim for 80)
- **Imports:** Grouped (stdlib, third-party, first-party) with blank lines between
- **Trailing commas:** Use in multi-line function signatures and collections
- **Docstrings:** Google format with Args/Returns/Raises sections

**Run before committing:**
```bash
pixi run fmt  # Black formatter
# TODO: Add pylint/pytype once configured
```

### Error Messages

**Guideline:** Error messages must be precise, greppable, and actionable.

```python
# ❌ BAD - Vague
if not path:
    raise ValueError("Path must be specified")

# ✅ GOOD - Specific and actionable
if not path:
    raise ValueError(
        f"vae_checkpoint_path must be specified in config. "
        f"Got config keys: {list(config.keys())}"
    )

# ✅ GOOD - Include available options
if step is None:
    available = mngr.all_steps() if hasattr(mngr, 'all_steps') else []
    raise ValueError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Available steps: {available if available else 'none'}"
    )
```

## Development Commands

### Environment Setup
```bash
# Install dependencies with pixi (recommended)
pixi install

# Activate environment
pixi shell

# For TPU environment
pixi shell -e tpu
```

### Code Formatting
```bash
# Format code with black
pixi run fmt

# Check formatting
black --check src/ scripts/
```

### Running Commands

**Rule:** Always use `pixi run` to run everything. This ensures reproducibility and correct environment.

```bash
# ✅ CORRECT - Always use pixi run
pixi run python scripts/train.py 

# ❌ WRONG - Never use bare python
python scripts/train.py
```

## Debugging Tactics

### General Principle

**Ground debugging in experimentation, avoid guessing.** Use empirical measurements (profiling, timing, memory stats) rather than assumptions about performance or behavior.

### Debugging Strategies

**1. Print Debugging for Interactive Experimentation**

Useful during initial development to understand data flow:

```python
# During development - OK for exploration
def prototype_function(inputs):
    print(f"Input shape: {inputs.shape}")
    result = complex_transform(inputs)
    print(f"Output shape: {result.shape}")
    return result
```

**Before committing:** Either remove or convert to logging:

```python
# After development - production ready
import logging
logger = logging.getLogger(__name__)

def prototype_function(inputs):
    logger.debug(f"Input shape: {inputs.shape}")
    result = complex_transform(inputs)
    logger.debug(f"Output shape: {result.shape}")
    return result
```

**2. Latency Testing**

Measure per-operation timing for performance bottlenecks:

```python
import time

# During development/profiling
start = time.perf_counter()
output = expensive_operation(input)
elapsed = time.perf_counter() - start
logger.info(f"Operation took {elapsed:.3f}s")

# Production: Keep behind verbose flag
if config.get("profile_latency", False):
    start = time.perf_counter()
    output = expensive_operation(input)
    logger.info(f"Operation took {time.perf_counter() - start:.3f}s")
```

**3. Memory Profiling**

Track device memory usage to debug OOM errors:

```python
def log_device_memory(prefix: str, verbose: bool = False) -> None:
    """Log device memory stats if verbose mode enabled."""
    if not verbose:
        return

    device = jax.devices()[0]
    if hasattr(device, 'memory_stats'):
        stats = device.memory_stats()
        used = stats.get("bytes_in_use", 0) / 1e9
        total = stats.get("bytes_limit", 0) / 1e9
        logger.info(f"[{prefix}] Memory: {used:.2f}GB / {total:.2f}GB")

# Usage - behind verbose flag
verbose = config.get("verbose_memory_logging", False)
log_device_memory("After model init", verbose=verbose)
```

**4. JAX Profiling**

Use JAX's built-in profiling for detailed performance analysis:

```python
# For detailed profiling sessions
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # Code to profile
    output = model(batch, rngs)
```

### Debugging Checklist Before Committing

- [ ] Remove all debug `print()` statements
- [ ] Convert essential prints to `logger.debug()`
- [ ] Put profiling/timing code behind verbose flags
- [ ] Remove temporary breakpoints or inspection code
- [ ] Ensure logging uses appropriate levels (DEBUG, INFO, WARNING, ERROR)

**Training Step Flow:**
1. Get batch from dataloader
2. Update training RNG key
3. `train_step(graphdef, state, optimizer_state, batch, train_key)` - JIT-compiled
   - Merge graphdef + state → model
   - Forward pass: `model(batch, rngs)`
   - Compute loss: `model.loss(batch, outputs)`
   - Compute gradients via `jax.grad`
   - Update optimizer state
   - Split model → (graphdef, state)
4. Log metrics (process 0 only)
5. Periodic evaluation and checkpointing

**Checkpoint Structure:**
```
checkpoints/{model_name}_{timestamp}/
├── step_1000/
│   ├── default/           # Model state (parameters)
│   ├── optimizer/         # Optimizer state (Adam moments)
│   └── metadata/          # Training metadata (step, config)
```

### Checkpoint Zoo Management

**Principle:** Centralize canonical checkpoint references to avoid fragmentation.

## Testing & Script Organization

### Test File Organization

**Rule:** Test files belong in `tests/` directory, not `scripts/`.

```
# ✅ GOOD structure
tests/
  test_latent_diffusion_api.py
  test_vae_loading.py
  test_dataloader_performance.py

scripts/
  train.py
  interactive_inference.py
  extract_inference_checkpoint.py

# ❌ BAD - Tests mixed with scripts
scripts/
  train.py
  test_latent_diffusion_api.py  # Should be in tests/
  interactive_inference.py
```

**Script categories:**
- `scripts/train.py` - Main training entrypoint
- `scripts/interactive_*.py` - User-facing interactive tools
- `scripts/extract_*.py` - Checkpoint utilities
- `scripts/visualize_*.py` - Data visualization tools
- `tests/test_*.py` - Unit and integration tests


### API Design Conventions

**Principle:** Function names should clearly indicate their behavior.

**Examples from this codebase:**

```python
# ❌ AMBIGUOUS - What kind of inference?
def inference(self, inputs, rngs):
    # Actually does teacher forcing with ground truth next frames
    ...

# ✅ CLEAR - Indicates teacher forcing
def teacher_forcing_inference(self, inputs, rngs):
    """Inference using teacher forcing (requires ground truth next frames)."""
    ...

# ✅ CLEAR - Indicates autoregressive generation
def autoregressive_inference(self, inputs, rngs):
    """Autoregressive generation (no ground truth needed)."""
    ...
```

**Common patterns:**
- `encode_*` / `decode_*` - Clearly paired operations
- `_load_*` - Loading from disk/checkpoint (private helper)
- `compute_*` - Pure computation (JIT-friendly)
- `log_*` - Side effects (logging, I/O)
- `*_step` - Single iteration of a loop


## Model Quality Recommendations

**For typical model scales (<500M parameters):**

2. **Memory-efficient settings:**
   - `dtype: bfloat16` for activations (2x memory reduction)
   - `param_dtype: float32` for parameters (stability)

3. **Typical hyperparameters:**
   - `model_dim: 512` (or 256 for smaller models)
   - `learning_rate: 1e-4` (scale up for larger batches)

## Common Gotchas

### Module Docstrings vs ABOUTME Comments

**Current pattern:** Files start with two `# ABOUTME:` comments.

**Recommended pattern:** Use comprehensive module docstrings instead.

```python
# ❌ Current pattern (being deprecated)
# ABOUTME: Test that LatentDiffusionWorldModel API matches PixelDiffusionWorldModel
"""Test that LatentDiffusionWorldModel API matches PixelDiffusionWorldModel."""

# ✅ Recommended pattern
"""Test LatentDiffusionWorldModel API consistency.

Verifies that LatentDiffusionWorldModel implements the ModelTrainerAPI
interface correctly and produces expected output shapes and dtypes.

This test ensures that the latent diffusion model can be used as a drop-in
replacement for the pixel diffusion model in the training pipeline.

Typical usage:
    python tests/test_latent_diffusion_api.py
"""
```

**Transition plan:**
- New files: Use comprehensive docstrings (no ABOUTME)
- Existing files: Gradually migrate ABOUTME → docstrings during refactors
- No need for mass migration, handle organically over time

**Rationale:** Docstrings are standard Python, show up in `help()`, and allow for richer documentation than single-line comments.
