import wandb


class WandbLogger:
    def __init__(self, project=None, run_name=None, config=None):
        wandb.login(key="f272b19ef214a7eec6c522fa7461a7d5fdec3e6a")
        self.run = wandb.init(project=project, name=run_name, config=config)

    def log(self, data, step=None, **kwargs):
        wandb.log(data, step=step, **kwargs)

    def image(self, path, **kwargs):
        return wandb.Image(path, **kwargs)

    def watch(self, model, **kwargs):
        wandb.watch(model, **kwargs)

    def finish(self):
        wandb.finish()
