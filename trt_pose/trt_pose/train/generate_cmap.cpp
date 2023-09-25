#include "generate_cmap.hpp"

namespace trt_pose {
namespace train {

torch::Tensor generate_cmap(torch::Tensor counts, torch::Tensor peaks, int height, int width, float stdev, int window)
{
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    
    int N = peaks.size(0);
    int C = peaks.size(1);
    int M = peaks.size(2);
    int H = height;
    int W = width;
    
    auto cmap = torch::zeros({N, C, H, W}, options);
    auto cmap_a = cmap.accessor<float, 4>();
    auto counts_a = counts.accessor<int, 2>();
    auto peaks_a = peaks.accessor<float, 4>();
    float var = stdev * stdev;

    // keep the same window/stdev ratio. In the original code, it is 5
    float window_ratio = static_cast<float>(window) / stdev;
    
    for (int n = 0; n < N; n++)
    {
        // n is the current batch
        for (int c = 0; c < C; c++)
        {
            // c is the current person/object
            int count = counts_a[n][c];
            for (int p = 0; p < count; p++)
            {
                // p is the current part (in person, it's right/left hip, shoulder, etc.)
                float i_mean = peaks_a[n][c][p][0] * H;
                float j_mean = peaks_a[n][c][p][1] * W;

                // enable variance per point, if tensor is the right size
                float current_std = -1;
                if(peaks_a.sizes()[3] > 2)
                    current_std = peaks_a[n][c][p][2];
                float current_var = var;
                int current_window = window;
                if(current_std >= 0 && current_std <= 1) {
                    current_std *= W;
                    current_var = current_std*current_std;
                    current_window = static_cast<int>(current_std * window_ratio);
                }
                // recompute everytime because of the possible change
                int w = current_window / 2;

                int i_min = i_mean - w;
                int i_max = i_mean + w + 1;
                int j_min = j_mean - w;
                int j_max = j_mean + w + 1;
                if (i_min < 0) i_min = 0;
                if (i_max >= H) i_max = H;
                if (j_min < 0) j_min = 0;
                if (j_max >= W) j_max = W;
                
                for (int i = i_min; i < i_max; i++)
                {
                    float d_i = i_mean - ((float) i + 0.5);
                    float val_i = - (d_i * d_i);
                    for (int j = j_min; j < j_max; j++)
                    {
                        float d_j = j_mean - ((float) j + 0.5);
                        float val_j = - (d_j * d_j);
                        float val_ij = val_i + val_j;
                        float val = expf(val_ij / current_var);
                        
                        if (val > cmap_a[n][c][i][j])
                        {
                            cmap_a[n][c][i][j] = val;
                        }
                    }
                }
            }
        }
    }
    
    return cmap;
}

} // namespace trt_pose::train
} // namespace trt_pose
