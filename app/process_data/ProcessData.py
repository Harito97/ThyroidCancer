import os
from app.process_data.PatchLevel import PatchLevel
from app.process_data.ClusterLevel import ClusterLevel


class ProcessData:
    """
    Return cropped images by 2 way crop (patch level & cluster level)
    Data will be save with label in data folder by default
    """

    def __init__(self) -> None:
        # change to home project
        ProcessData.cd_home()

        # load model
        self.patch_level = PatchLevel()
        self.cluster_level = ClusterLevel()

    def patch_level_crop(
        self,
        data_folder: str = "data/origin_data/B256/",
        destination: str = "data/processed_data/patch_level/",
    ):
        for folder in os.listdir(data_folder):
            for image_id in os.listdir(data_folder + folder):
                cropped_images = self.patch_level.process(
                    img_path=data_folder + folder + "/" + image_id
                )
                PatchLevel.save_result(
                    images=cropped_images,
                    label=folder,
                    img_origin_id=image_id,
                    destination=destination,
                )

    def cluster_level_crop(self): ...

    def cd_home():
        cwd = os.getcwd()
        if cwd[-13:] != "ThyroidCancer":
            os.chdir(cwd[:-16])  # change to home project
