class ClusterLevel:
    """
    Step 1. Detect one biggest rectangle contain all cell in picture
    Step 2. Identify the center then slice the rectangle to 224x224 windows
    Step 3. With each sliced 224x224 window -> re-segment: is contain enough cell -> select or not
    """

    def __init__(self) -> None:
        pass

    # def process_mask_and_plot(self):
    #     final_result = []
            
    #     for result in self.results:
    #         # Lấy ra các mask tương ứng trong 1 ảnh 
    #         mask_array = result.masks.xy
    #         black_image = np.zeros((768, 1024))
    #         for mask in mask_array:
    #             cv2.fillPoly(black_image, pts=np.int32([mask.reshape((-1, 1, 2))]), color=255) 

    #         # Tính tổng màu trắng trong mỗi ô 256x256
    #         data = []
    #         for i in range(3):
    #             for j in range(4):
    #                 sum_white_pixels = np.sum(black_image[i*256:(i+1)*256, j*256:(j+1)*256] == 255)
    #                 data.append(sum_white_pixels)


    #         # ***Lấy chỉ số của 5 ô có tổng số màu trắng lớn nhất***
    #         sorted_indices = np.argsort(data)[::-1][:5]

    #         image = result.plot(conf=False,
    #                             labels=False,
    #                             boxes=False,
    #                             masks=True,
    #                             probs=False)
            
    #         image = CropImage_DeepLearning.__draw_bounding_boxes__(image, sorted_indices)

    #         final_result.append(image)

    #     return final_result
