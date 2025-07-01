import wandb


class WandbLogger:
    def __init__(self, project=None, run_name=None, config=None):
        wandb.login(key="f272b19ef214a7eec6c522fa7461a7d5fdec3e6a")
        self.run = wandb.init(project=project, name=run_name, config=config)

    def log(self, data, step=None):
        wandb.log(data, step=step)

    def image(self, path):
        return wandb.Image(path)

    def watch(self, model):
        wandb.watch(model)

    def finish(self):
        wandb.finish()
