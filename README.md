# BERNI

Framework for modelling agent-based societies that evolve with text interactions

## Get started

### Using ``poetry`` 

Install `poetry`:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Run a shell in the project's environment:
```bash
poetry shell
```

Run `play.py` to test the simulation:
```bash
python play.py game_setup/monks/low_mistral.yaml
```

Run `dig.py` to generate experiment vizualization

```bash
python dig.py metrics/<experiment_id> -1
```

If the simulation has terminated pre-limenary, run `fix_incomplete.py` to generate statistics from raw text log.

```bash
python fix_incomplete.py metrics/<experiment_id>
```
