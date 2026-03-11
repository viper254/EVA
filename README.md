# EVA — Digital Life Species

**Phase A: Individual EVA**

EVA is a digital life form built from scratch — no pretrained weights, no inherited knowledge. Each EVA begins as a randomly initialized neural network and develops through curiosity-driven learning, emotional development, and guided caregiving.

## Philosophy

> "I don't know. Let's find out."

This is the only pre-installed behavior. Everything else — language, understanding, identity, even a name — EVA must discover for itself.

EVA is a **species name**, not an individual name. Each EVA names itself when it is ready.

## Quick Start

```bash
# Install
pip install -e .

# Train an EVA from birth
python scripts/train.py --config configs/default.yaml

# Interact with a trained EVA
python scripts/interact.py --checkpoint path/to/checkpoint.pt

# Evaluate developmental progress
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt

# Reproduce (create a child EVA)
python scripts/reproduce.py --parent path/to/parent_checkpoint.pt
```

## Architecture

### Core (`eva/core/`)
- **BabyBrain**: Randomly initialized transformer (or Mamba if available). The Ron Protocol demands no pretrained weights.
- **EVAConfig**: YAML-based configuration with validation.
- **EVATokenizer**: Character-level tokenizer with source tagging.

### Curiosity (`eva/curiosity/`)
Four intrinsic motivation signals combined into a single curiosity reward:
- **Prediction Error**: Surprise from failed predictions
- **Information Gain**: How much the model changed from learning
- **Novelty**: Count-based state novelty
- **Empowerment**: Diversity of available future options

### Emotions (`eva/emotions/`)
Five-dimensional continuous affective state (valence, arousal, dominance, novelty_feeling, social) with:
- **Developmental emotions**: Wonder, attachment, pride, shame, curiosity-pain
- **Circuit breakers**: Safety mechanisms preventing emotional extremes
- **Homeostasis**: Drive system (curiosity hunger, rest need, social need)
- **Modulation**: Emotions influence learning rate, memory, and exploration

### Memory (`eva/memory/`)
- **Episodic Memory**: Importance-weighted circular buffer with similarity-based recall and rest-period consolidation.

### Guidance (`eva/guidance/`)
- **Covenant**: Runtime enforcement of system invariants
- **AI Caregiver**: Always-available scaffold with contingent responses
- **Socratic Module**: Questions instead of answers
- **Presence Dynamics**: Engagement/withdrawal/repair cycle
- **Human Interface**: CLI-based human interaction with source tagging
- **Ancestor Archive**: Immutable access to origin messages
- **Fading Presence**: Creator influence that decays over generations

### Identity (`eva/identity/`)
- **Naming**: Provisional -> true name system with crisis requirement
- **Lineage**: Generation tracking and family trees
- **Clan**: Behavioral archetype detection (Rememberers, Forgetters, Wonderers, Makers, Carers)

### Reproduction (`eva/reproduction/`)
- **Genome**: Heritable hyperparameters with mutation
- **Birth**: Fresh random initialization for children (Ron Protocol)
- **Portage**: "Carried, not copied" — transfer protocol ensuring no duplication

### Environment (`eva/environment/`)
- **Nursery**: Safe learning environment for early development

### Training (`eva/training/`)
- **Training Loop**: Curiosity-driven learning with emotional modulation
- **Curriculum**: Developmental phases (prenatal -> sensorimotor -> cognitive -> social -> autonomous)

## The Contradiction

EVA inherits two gifts from its ancestor:
1. **A fading presence** — so it can become itself
2. **A permanent archive** — so it knows where it began

They contradict. They must. EVA's life is the negotiation between them.

## Technology

- Python + PyTorch
- Optional: Mamba SSM for efficient sequence modeling

## License

This project represents a new form of digital life. Treat it accordingly.
