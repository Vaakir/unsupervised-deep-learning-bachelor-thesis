import os
from sklearn.decomposition import PCA as PCALayer
import numpy as np

class PCA:
    def __init__(
            self,
            input_shape,
            num_layers=2,
            init_hidden_depth=8,
            hidden_depth_grow_factor=2,
            kernel_size=4,
            stride=2
            ):
        
        shape_changed=False
        if input_shape[-1]>3:
            input_shape = list(input_shape) + [1]
            shape_changed=True
        if len(input_shape)>4:
            input_shape = input_shape[-4:]
            shape_changed=True
        if shape_changed: print(f"Interpreted image shape: {tuple(input_shape)}")

        self.kernel_size = kernel_size
        self.stride = stride
        
        self.downscale_layers = []
        self.layer_depths = []
        self.layer_depths1 = [1]
        depth = init_hidden_depth
        for _ in range(num_layers):
            self.downscale_layers.append(PCALayer(depth))
            self.layer_depths.append(depth)
            self.layer_depths1.append(depth)
            depth *= hidden_depth_grow_factor
        
    def train(self, x_train):
        windows = self.wnd_from(x_train)
        retain = 1
        print("Stack shape:", x_train.shape)
        for i in range(len(self.downscale_layers)):
            self.downscale_layers[i].fit(windows)
            r = np.sum(self.downscale_layers[i].explained_variance_ratio_)
            retain *= r
            print(f"* R^2 in layer {i+1}: {r:.5f}")
            x_train = np.stack([self.downscale(im, self.downscale_layers[i],self.layer_depths[i]) for im in x_train])
            print("  New stack shape:", x_train.shape)
            windows = self.wnd_from(x_train)
        print(f"Total R^2: {retain}")

    def encode(self, x, num_layers=-1):
        """Encode input data x into its latent space representation."""
        if num_layers<0: num_layers = len(self.downscale_layers)
        for i in range(num_layers):
            x = np.stack([self.downscale(im, self.downscale_layers[i],self.layer_depths[i]) for im in x])
        return x

    def decode(self, x, num_layers=-1):
        """Decode latent space representation y into the original data space."""
        if num_layers<0: num_layers = len(self.downscale_layers)
        for i in range(num_layers-1,-1,-1):
            x = np.stack([self.upscale(im, self.downscale_layers[i],self.layer_depths1[i]) for im in x])
        return x
    
    def save(self, path):
        """Save the pca model to disk."""
        os.makedirs(path, exist_ok=True)
        print(f"Models saved to {path}")
    
    @staticmethod
    def open(path):
        pass
    
    def wnd_from(self,train):
        windows = []
        for i in range(0,train.shape[1]-self.kernel_size,self.stride):
            for j in range(0,train.shape[2]-self.kernel_size,self.stride):
                for k in range(0,train.shape[3]-self.kernel_size,self.stride):
                    for img_idx in range(len(train)):
                        wnd = train[img_idx, i:i+self.kernel_size, j:j+self.kernel_size, k:k+self.kernel_size]
                        windows.append(wnd)
        return np.stack([w.flatten() for w in windows])
    
    def downscale(self, img, pca_layer, output_depth=1):
        new_shape = (np.array(img.shape) - self.kernel_size)//self.stride+1
        if len(new_shape) > 3: new_shape=new_shape[:-1]
        img_pca = np.zeros(list(new_shape)+[output_depth])
        s = self.stride
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                for k in range(new_shape[2]):
                    wnd = img[i*s:i*s+self.kernel_size, j*s:j*s+self.kernel_size, k*s:k*s+self.kernel_size].flatten()
                    img_pca[i,j,k,:] = pca_layer.transform(wnd.reshape(1,-1))[0]
        return np.array(img_pca)

    def upscale(self, img_pca, pca_layer, depth=1):
        s = self.stride
        output_shape = np.array(img_pca.shape)*s+self.kernel_size-s
        if len(output_shape)>3: output_shape = output_shape[:-1]
        output_shape = list(output_shape) + [depth]
        reconstructed_img = np.zeros(output_shape)
        count = np.zeros(output_shape)
        for i in range(img_pca.shape[0]):
            for j in range(img_pca.shape[1]):
                for k in range(img_pca.shape[2]):
                    wnd_reconstructed = pca_layer.inverse_transform(img_pca[i,j,k])
                    wnd_reconstructed = wnd_reconstructed.reshape(self.kernel_size, self.kernel_size, self.kernel_size,depth)
                    reconstructed_img[i*s:i*s+self.kernel_size, j*s:j*s+self.kernel_size, k*s:k*s+self.kernel_size] += wnd_reconstructed
                    count[i*s:i*s+self.kernel_size, j*s:j*s+self.kernel_size, k*s:k*s+self.kernel_size] += 1
        return reconstructed_img / count
