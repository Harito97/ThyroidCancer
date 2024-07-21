import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import wandb


class Tool:
    @staticmethod
    def save_confusion_matrix(y_true, y_score, target_names, filename, normalize=False):
        """
        Saves the confusion matrix to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            target_names (list): Names of the target classes.
            filename (str): Path to save the confusion matrix.
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        Returns:
            cm (numpy.ndarray): The confusion matrix.
        """
        try:
            cm = confusion_matrix(y_true, y_score)
            print("Confusion Matrix:\n", cm)
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                title = "Normalized Confusion Matrix"
            else:
                title = "Confusion Matrix, Without Normalization"

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names,
            )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.title(title)
            plt.savefig(filename)
            plt.close()
            return cm
        except ValueError as e:
            print(f"Error creating confusion matrix: {e}")
            return None

    @staticmethod
    def save_classification_report(y_true, y_score, filename):
        """
        Saves the classification report to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            filename (str): Path to save the classification report.
        Returns:
            cr (dict): The classification report.
        """
        try:
            cr = classification_report(y_true, y_score, output_dict=True)
            print("Classification Report:\n", cr)
            report_df = pd.DataFrame(cr).transpose()
            report_df.drop(
                "support", axis=1, inplace=True
            )  # Bỏ cột support nếu không cần
            report_df.plot(kind="bar", figsize=(10, 6))
            plt.title("Classification Report")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return cr
        except ValueError as e:
            print(f"Error creating DataFrame from classification report: {e}")
            return None

    @staticmethod
    def save_roc_auc_plot(y_true, y_score, n_classes, filename):
        """
        Calculates and saves the ROC AUC plot to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            n_classes (int): Number of classes.
            filename (str): Path to save the plot.
        Returns:
            fpr (dict): False positive rates for each class.
            tpr (dict): True positive rates for each class.
            roc_auc (dict): ROC AUC scores for each class.
        """
        try:
            # Convert y_true and y_score to NumPy arrays if they are lists
            y_true = np.array(y_true)
            y_score = np.array(y_score)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Binarize the output if more than 2 classes
            if n_classes > 2:
                y_true = label_binarize(y_true, classes=[*range(n_classes)])
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[1], tpr[1], _ = roc_curve(y_true, y_score[:, 1])
                roc_auc[1] = auc(fpr[1], tpr[1])

            plt.figure(figsize=(8, 6))

            if n_classes == 2:
                plt.plot(
                    fpr[1],
                    tpr[1],
                    lw=2,
                    label="ROC curve (area = {0:0.2f})".format(roc_auc[1]),
                )
            else:
                for i in range(n_classes):
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        lw=2,
                        label="ROC curve of class {0} (area = {1:0.2f})".format(
                            i, roc_auc[i]
                        ),
                    )

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.savefig(filename)
            plt.close()
            return fpr, tpr, roc_auc
        except ValueError as e:
            print(f"Error creating ROC AUC plot: {e}")
            return None, None, None


class TestClassification:
    def __init__(
        self,
        test_loader,
        model,
        device,
        criterion,
        model_destination=".",
        model_name="model",
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.criterion = criterion
        self.model_destination = model_destination
        self.model_name = model_name

    def load_model(self):
        try:
            print("Loading model weights...")
            model_path = os.path.join(
                self.model_destination, f"best_{self.model_name}_model.pt"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Set model to evaluate mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def run_test(self):
        try:
            # Initialize W&B
            wandb.init(
                project="ThyroidCancer",
                entity="harito",
                settings=wandb.Settings(start_method="thread"),
            )

            # Test loop
            test_preds, test_targets, test_probs = [], [], []
            total_loss = 0

            with torch.no_grad():
                total_batches = len(self.test_loader)  # Get total number of batches

                for i, (images, labels) in enumerate(self.test_loader):
                    progress = (i + 1) / total_batches * 100  # Calculate progress
                    print(f"\rProgress: {progress:.2f}%", end="")

                    # Load data of 1 batch to device
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    test_preds.extend(preds.view(-1).cpu().numpy())
                    test_targets.extend(labels.view(-1).cpu().numpy())
                    test_probs.extend(outputs.cpu().numpy())  # Save probabilities

            # Calculate metrics
            test_loss = total_loss / len(self.test_loader)
            test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
            test_f1 = f1_score(test_targets, test_preds, average="weighted")

            print(
                f"\nTest Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}, Test F1: {test_f1:.6f}"
            )

            # Log metrics to W&B
            wandb.log(
                {"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1}
            )

            # Confusion Matrix and Classification Report
            unique_labels = np.unique(test_targets)
            target_names = [str(label) for label in unique_labels]
            cm = Tool.save_confusion_matrix(
                y_true=test_targets,
                y_score=test_preds,
                target_names=target_names,
                filename=f"{self.model_destination}/confusion_matrix.png",
            )
            cr = Tool.save_classification_report(
                y_true=test_targets,
                y_score=test_preds,
                filename=f"{self.model_destination}/classification_report.png",
            )
            fpr, tpr, roc_auc = Tool.save_roc_auc_plot(
                y_true=test_targets,
                y_score=test_probs,
                n_classes=len(unique_labels),
                filename=f"{self.model_destination}/roc_auc_plot.png",
            )

            # Log additional metrics and plots to W&B
            try:
                wandb.log(
                    {
                        "confusion_matrix": wandb.Image(
                            f"{self.model_destination}/confusion_matrix.png"
                        ),
                        "classification_report": wandb.Image(
                            f"{self.model_destination}/classification_report.png"
                        ),
                        "roc_auc_plot": wandb.Image(
                            f"{self.model_destination}/roc_auc_plot.png"
                        ),
                    }
                )
            except Exception as e:
                print(f"Error logging metrics to W&B: {e}")

            # Save test information to npz file
            print(
                f"Saving test metrics to {os.path.join(self.model_destination, f'test_{self.model_name}_metrics.npz')}"
            )
            np.savez(
                os.path.join(
                    self.model_destination, f"test_{self.model_name}_metrics.npz"
                ),
                test_preds=test_preds,
                test_probs=test_probs,
                test_targets=test_targets,
                test_loss=test_loss,
                test_acc=test_acc,
                test_f1=test_f1,
                target_names=target_names,
                cm=cm,
                cr=cr,
                roc_auc=roc_auc,
                fpr=fpr,
                tpr=tpr,
            )
            print("Test completed")
        except Exception as e:
            print(f"Error during testing: {e}")
        finally:
            wandb.finish()


# Usage
def test(
    test_loader=None,
    model=None,
    device=None,
    criterion=None,
    model_destination=".",
    model_name="model",
):
    if not all([test_loader, model, device, criterion]):
        print(
            "Missing one or more required arguments: test_loader, model, device, criterion"
        )
        return

    tester = TestClassification(
        test_loader=test_loader,
        model=model,
        device=device,
        criterion=criterion,
        model_destination=model_destination,
        model_name=model_name,
    )
    tester.load_model()
    tester.run_test()
