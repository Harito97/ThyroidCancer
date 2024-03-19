import os, cv2, numpy
from ultralytics import YOLO
from sklearn.cluster import KMeans


class PatchLevel:
    """
    + Step 1. Detect / Segment all cell
    + Step 2. K-mean / Density cluster -> Identify all center (select max 5 center) -> Become a patch
    + Step 3. Use re-segment -> Remove the patch doesn't have enough number of cell in
    """

    def __init__(self) -> None:
        # change to home project
        PatchLevel.cd_home()

        # load model
        self.model = PatchLevel.load_model()

    def process(
        self,
        img_path: str = None,
        conf: float = 0.8,
        num_clusters: int = 5,
        crop_size: int = 224,
        threshold: int = 30,
        show_output: tuple = (False, False, False),
    ):
        """
        Para:
            see detail in doc of step1(), step2(), step3()
        Return:
            final result of step 3
        Warning:
            when use this function to pre-process data for training further
            -> should set show_output all False or at least show_output[1] be False
        """
        result = self.step1(img_path=img_path, conf=conf, show_output=show_output[0])
        threshold = len(result[0]) / 10
        result = self.step2(
            result=result,
            num_clusters=num_clusters,
            crop_size=crop_size,
            show_output=show_output[1],
        )
        return self.step3(
            result=result,
            conf=conf / 1.2,
            threshold=threshold,
            show_output=show_output[2],
        )

    def step1(self, img_path: str = None, conf: float = 0.8, show_output: bool = False):
        """
        Para:
            img_path: directory of the image
            conf: confidence to filter the cell
            show_df: choice to show output the fist detect or not
        Return:
            result: the result from yolo model - use as a parameter in step 2
        """
        # error when img_path null
        if img_path == None:
            print("Image path is None. Please put in a image path")
            return

        # read image and predict cell with confidence in it
        img = cv2.imread(img_path)
        result = self.model.predict(img, conf=conf)

        # show output if true
        if show_output == True:
            import matplotlib.pyplot as plt

            plt.imshow(result[0].plot())
            plt.show()

        return result

    def step2(
        self,
        result: list,
        num_clusters: int = 5,
        crop_size: int = 224,
        show_output: bool = False,
    ):
        """
        Para:
            result: the output of yolo_model.predict() in step 1
            num_clusters: number of clusters want when clustering
            crop_size: size of cropped image to fit for training model in classify task
            show_output: choice show output in this step
        Return:
            a list of cropped images with crop_size and contain wanted cells
        """
        # get origin image
        image = result[0].orig_img
        # get size of image
        size = result[0].orig_shape
        MIN_X, MIN_Y = crop_size / 2, crop_size / 2
        MAX_X, MAX_Y = size[1] - MIN_X, size[0] - MIN_Y  # x is width, y is height
        # get boxes with format is center x, center y, width, height
        xywh = result[0].boxes.xywh
        # get x center, y center
        x_centers = xywh[:, 0]
        y_centers = xywh[:, 1]
        # combine x center and y center to a 2D array
        x_centers = (
            x_centers.cpu().numpy()
        )  # fix to make sure can run on GPU of Google Colab
        y_centers = (
            y_centers.cpu().numpy()
        )  # fix to make sure can run on GPU of Google Colab
        centers = numpy.column_stack((x_centers, y_centers))
        # init KMeans
        kmeans = KMeans(n_clusters=min(len(centers), num_clusters))
        # cluster process
        kmeans.fit(centers)
        # get center of clusters
        cluster_centers = kmeans.cluster_centers_
        # round the cluster centers
        cluster_centers = numpy.round(cluster_centers).astype(int)
        # get bouding boxes from cluster centers
        bounding_boxes = []
        half_size = crop_size // 2
        for center in cluster_centers:
            x, y = center
            # make sure the bounding box only in the image
            x = min(max(x, MIN_X), MAX_X)
            y = min(max(y, MIN_Y), MAX_Y)
            x_min, y_min = x - half_size, y - half_size
            x_max, y_max = x + half_size, y + half_size
            bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        # get crop image
        crop_images = []
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            crop_images.append(image[y_min:y_max, x_min:x_max])
        # show output if true
        if show_output == True:
            import matplotlib.pyplot as plt

            for box in bounding_boxes:
                cv2.rectangle(
                    image,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color=(0, 255, 0),
                    thickness=5,
                )
            plt.imshow(image)
            plt.show()
        return crop_images

    def step3(
        self,
        result: list,
        conf: float = 0.8,
        threshold: int = 30,
        show_output: bool = False,
    ):
        """
        Para:
            result: result of step 2
        Return:
            a list of cropped images that good enough
        """
        final_result = []
        for image in result:
            _ = self.model.predict(image, conf=conf)
            if len(_[0]) > threshold:
                final_result.append(image)
        # show output if true
        if show_output == True:
            import matplotlib.pyplot as plt

            for image in final_result:
                plt.imshow(image)
                plt.show()
        return final_result

    """
    Functions of class
    """

    def cd_home():
        cwd = os.getcwd()
        if cwd[-13:] != "ThyroidCancer":
            os.chdir(cwd[:-16])  # change to home project

    def load_model(model: str = "model/process_data/detect/cell_detect_ver1.pt"):
        """
        Load in the model use to detect (or segment) with model parameter is the directory path to the model
        """
        if model == "":
            print("There is no model path")
            return -1
        return YOLO(model)

    def save_result(
        images: list,
        label: str,
        img_origin_id: str,
        destination: str = "data/processed_data/patch_level/",
    ):
        """
        Para:
            images: the result of process() or step3()
            label: the label of the processing image
            img_origin_id: the id of the processing image
            destination: the destination to save cropped image
        """
        for i, image in enumerate(images):
            cv2.imwrite(destination + label + "_" + str(i) + "_" + img_origin_id, image)
