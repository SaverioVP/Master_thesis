class DetectorCalculator():       
    def __init__(self, size_px = 256, size_mm = 14.1):
        """
        The detector doesn't actually exist. Only these parameters are relevant
        """
        self.size_px = size_px  # N, pixel size of detector (NxN): always square
        self.size_mm = size_mm  # height of detector
        self.resolution_mm = self.size_mm / self.size_px  # pixel resolution


