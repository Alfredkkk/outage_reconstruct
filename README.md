# Pipeline Running
Please refer to `EXPERIMENT_REPRODUCTION.md` for guidance. It's not much different from the `GROWER M&V Pipeline Meta Doc.pdf`. I modified some parameters, but nothing too significant. This repo serves as our own project repo.

The current ground-truth builder and its audit outputs are documented in
`GROUND_TRUTH_CHANGES.md`. A run made with the current source uses ground-truth
version 2.0 and is therefore a corrected experiment, not an exact regeneration
of the submitted legacy ground truth.
# Old One-Click method by Howshing
1. `pip install rich-click`
## How to run
1. `python mvp_handler.py -c {config file} -a {algo name}`. 
2. ex: `python mvp_handler.py -c mvp_config.yaml -a mixed_threshold`
## Instructions
1. please find the todo sections
2. should see something like this in the output file:
    ![alt text](image.png)
