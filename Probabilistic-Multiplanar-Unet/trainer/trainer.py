class Trainer():

    def predict(self, imgs, masks):
        raise NotImplementedError

    def eval(self, imgs, true_masks, masks_pred):
        raise NotImplementedError

    def loss(self, imgs, true_masks, masks_pred):
        raise NotImplementedError

    def mask_to_image(self, mask, prediction=False):
        raise NotImplementedError