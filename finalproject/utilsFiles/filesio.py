import os


class FilesIO:

    """
    这是一个文件IO流类
    用于获取图片保存路径
    """

    @staticmethod
    def getSavePath(name: str):

        """
        获取图片保存路径
        """

        cur_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(os.path.dirname(cur_path))
        save_path = os.path.join(cur_dir, "images", name)
        return save_path


if __name__ == "__main__":

    print(FilesIO.getSavePath("image_1.png"))