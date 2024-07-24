from dataclasses import dataclass
import numpy as np

@dataclass
class GazeResultContainer:

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
    # index_to_results: list #see pipeline for details

    def sort_by_x1(self):
        """
        sort all data by the indicies of the x1-coord sorting of bboxes
        """
        x1_coords = self.bboxes[:,0]
        sorting_indices = np.argsort(x1_coords)
        self.pitch = self.pitch[sorting_indices]
        self.yaw = self.yaw[sorting_indices]
        self.bboxes = self.bboxes[sorting_indices]
        self.landmarks = self.landmarks[sorting_indices]
        self.scores = self.scores[sorting_indices]

    def top_n(self, n):
        """
        Keeps only the top n faces sorted by bbox area.
        :param n: Number of faces to keep
        """
        # Calculate the area of each bounding box
        widths = self.bboxes[:, 2] - self.bboxes[:, 0]
        heights = self.bboxes[:, 3] - self.bboxes[:, 1]
        areas = widths * heights

        # Get the indices of the top n largest areas
        top_n_indices = np.argsort(areas)[-n:][::-1]

        # Select the top n elements based on these indices
        self.pitch = self.pitch[top_n_indices]
        self.yaw = self.yaw[top_n_indices]
        self.bboxes = self.bboxes[top_n_indices]
        self.landmarks = self.landmarks[top_n_indices]
        self.scores = self.scores[top_n_indices]


def testing():
    """
    Test to show bboxes can be ordered and filtered
    """
    pitch = np.array([-0.03366609, -0.8437219, 0.00309433, -0.22087248], dtype=np.float32)
    yaw = np.array([0.3044246, -0.24106988, 0.7015595, 1.6160532], dtype=np.float32)
    bboxes = np.array([
        [1679.1862, 230.25752, 1741.6626, 304.60083],
        [1514.8796, 342.2738, 1651.6279, 508.54785],
        [1031.8713, 39.447083, 1099.9047, 130.67342],
        [722.9281, 482.517, 898.4352, 727.7577]
    ], dtype=np.float32)
    landmarks = np.array([
        [[1697.7335, 255.45403],
         [1726.2806, 257.91617],
         [1710.3767, 270.3583],
         [1697.4132, 283.70422],
         [1719.9648, 285.77567]],
        [[1557.7438, 411.42984],
         [1616.751, 433.59668],
         [1576.1996, 465.31885],
         [1541.1897, 466.37695],
         [1588.5776, 484.65106]],
        [[1062.9617, 78.44956],
         [1088.3052, 74.47197],
         [1087.5742, 88.829926],
         [1072.0139, 110.35148],
         [1090.4277, 107.41326]],
        [[840.0394, 597.48486],
         [856.9166, 587.40485],
         [882.67236, 644.6561],
         [845.2059, 684.1894],
         [853.3457, 676.4181]]
    ], dtype=np.float32)
    scores = np.array([0.9990476, 0.9988562, 0.9962057, 0.6899895], dtype=np.float32)

    # Create GazeResultContainer instance
    results = GazeResultContainer(pitch, yaw, bboxes, landmarks, scores)

    # Sort by the first x-coordinate in the bounding boxes
    results.sort_by_x1()

    # Print the sorted results
    print("UnSorted bboxes:", bboxes)
    print("UnSorted pitches:", pitch)
    print("UnSorted yaws:", yaw)
    print("Sorted bboxes:", results.bboxes)
    print("Sorted pitches:", results.pitch)
    print("Sorted yaws:", results.yaw)


    # Select the top n largest bounding boxes
    n = 2
    results.top_n(n)

    # Print the top n results
    print(f"\nTop {n} pitches:", results.pitch)
    print(f"Top {n} yaws:", results.yaw)
    print(f"Top {n} bboxes:", results.bboxes)
    print(f"Top {n} landmarks:", results.landmarks)
    print(f"Top {n} scores:", results.scores)

if __name__ == '__main__':
    testing()
