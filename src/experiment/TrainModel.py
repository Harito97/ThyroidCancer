import os
import json
import torch
import wandb
import numpy as np
from sklearn.metrics import f1_score


class TrainGradualUnfreezing:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        device,
        criterion,
        optimizer,
        num_epochs=100,
        patience=30,
        model_destination=".",
        model_name="model",
        n_layers_to_unfreeze=1,
        wandb_session_name="H97",
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
        self.model_destination = model_destination
        self.model_name = model_name
        self.n_layers_to_unfreeze = n_layers_to_unfreeze
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
        self.best_f1 = -1
        self.patience_counter = 0

        if not os.path.exists(model_destination):
            os.makedirs(model_destination)

        # Initialize W&B
        wandb.init(
            project="ThyroidCancer",
            entity="harito",
            config={
                "num_epochs": num_epochs,
                "patience": patience,
                "learning_rate": optimizer.defaults["lr"],
                "model_name": model_name,
                "n_layers_to_unfreeze": n_layers_to_unfreeze,
            },
            name=wandb_session_name,
            settings=wandb.Settings(start_method="thread"),
        )

    def _set_layer_requires_grad(self, n_layers):
        """
        Set the requires_grad attribute for layers based on the number of layers to unfreeze.
        """
        # Calculate which layers to unfreeze
        start_layer = max(0, self.model.num_layers - n_layers)
        self.model.set_parameter_requires_grad(start_layer, self.model.num_layers)

    def train(self):
        print("Moving model, criterion, optimizer to device ...")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        history_file_path = os.path.join(
            self.model_destination, f"{self.model_name}_history.json"
        )

        print("Training classification model...")
        for epoch in range(self.num_epochs):
            # Gradual unfreezing logic
            n_layers = (epoch + 1) * self.n_layers_to_unfreeze
            self._set_layer_requires_grad(n_layers)

            # Update optimizer for current layer setup
            self.optimizer = self.model.get_optimizers(
                self.optimizer.param_groups[0]["lr"]
            )

            # Training phase
            self.model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []

            total_batches = len(self.train_loader)  # Get total number of batches
            for i, (images, labels) in enumerate(self.train_loader):
                progress = (i + 1) / total_batches * 100  # Calculate progress
                print(f"\rProgress: {progress:.2f}%", end="")

                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.view(-1).cpu().numpy())
                train_targets.extend(labels.view(-1).cpu().numpy())

            train_loss = running_loss / len(self.train_loader)
            train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
            train_f1 = f1_score(train_targets, train_preds, average="weighted")

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.view(-1).cpu().numpy())
                    val_targets.extend(labels.view(-1).cpu().numpy())

            val_loss = val_running_loss / len(self.valid_loader)
            val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
            val_f1 = f1_score(val_targets, val_preds, average="weighted")

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            # Log metrics to W&B
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch": epoch + 1,
                }
            )

            # Save history
            with open(history_file_path, "w") as history_file:
                json.dump(self.history, history_file)

            # Checkpoint
            if val_f1 > self.best_f1 or (
                val_f1 == self.best_f1 and train_f1 > self.history["train_f1"][-2]
            ):
                self.best_f1 = val_f1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_destination, f"best_{self.model_name}_model.pt"
                    ),
                )
                print("\nSaved **best model** at epoch", epoch + 1)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_destination, f"last_{self.model_name}_model.pt"
                    ),
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                break

        print("Training completed")
        wandb.finish()
