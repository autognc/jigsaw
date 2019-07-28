class BoundingBox:
    """Stores the label and bounding box dimensions for a detected image region
    
    Attributes:
        label (str): the classification label for the region (e.g., "cygnus")
        xmin (int): the pixel location of the left edge of the bounding box
        xmax (int): the pixel location of the right edge of the bounding box
        ymin (int): the pixel location of the top edge of the bounding box
        ymax (int): the pixel location of the top edge of the bounding box
    """

    def __init__(self, label, xmin, xmax, ymin, ymax, label_int_class):
        self.label = label
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.label_int_class = label_int_class

    def __repr__(self):
        return "label: {} | xmin: {} | xmax: {} | ymin: {} | ymax: {}".format(
            self.label, self.xmin, self.xmax, self.ymin, self.ymax)
    
    @property
    def label_int(self):
        return self.label_int_class._label_to_int_dict[self.label]
        
class Transform:
    """Stores basic information regarding an image transformation

    Attributes:
        transform_type (str): the type of transform that this is: "rename" or
            "merge"
        original (str/list): the original label that should be renamed, or the
            list of labels that should be merged
        new (str): the output label name after a rename or merge
    """

    def __init__(self, transform_type, original, new):
        self.transform_type = transform_type
        self.original = original
        self.new = new

    def perform_on_image(self, labeled_image_mask):
        """Performs this transformation on a given LabeledImageMask object
        
        Args:
            labeled_image_mask (LabeledImageMask): an object representative of
                a semantically-labeled image
        """
        if self.transform_type == "rename":
            labeled_image_mask.rename_label(self.original, self.new)
        if self.transform_type == "merge":
            labeled_image_mask.merge_labels(self.original, self.new)
