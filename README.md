![image](https://github.com/user-attachments/assets/0a736fdd-6423-4777-b71c-11f8bcee9e29)<img align="right" src="https://ars.els-cdn.com/content/image/X00104825.jpg" width="290" height="350"/>  

# CRNN-Refined Spatiotemporal Transformer for Dynamic MRI Reconstruction  

* ### Abstract
Magnetic Resonance Imaging (MRI) plays a pivotal role in modern clinical practice, providing detailed anatomical visualization with exceptional spatial resolution and soft tissue contrast. Dynamic MRI, aiming to capture both spatial and temporal characteristics, faces challenges related to prolonged acquisition times and susceptibility to motion artifacts. Balancing spatial and temporal resolutions becomes crucial in real-world clinical scenarios. In the realm of dynamic MRI reconstruction, while Convolutional Recurrent Neural Networks (CRNNs) struggle with long-term dependencies, RNNs require extensive iterations, impacting efficiency. Transformers, known for their effectiveness in high-dimensional imaging, are underexplored in dynamic MRI reconstruction. Additionally, prevailing algorithms fall short of achieving superior results in demanding generative reconstructions at high acceleration rates. This research proposes a novel approach for dynamic MRI reconstruction, named CRNN-Refined Spatiotemporal Transformer Network (CST-Net). The spatiotemporal Transformer initiates reconstruction, modeling temporal and spatial correlations, followed by refinement using the CRNN. This integration mitigates inaccuracies caused by damaged frames and reduces CRNN iterations, enhancing computational efficiency without compromising reconstruction quality. Our study compares the performance of the proposed CST-Net at 6× and 12× undersampling rates, showcasing its superiority over existing algorithms. Particularly, in challenging 25× generative reconstructions, the CST-Net outperforms current methods. The comparison includes experiments under both radial and Cartesian undersampling patterns. In conclusion, CST-Net successfully addresses the limitations inherent in existing generative reconstruction algorithms, thereby paving the way for further exploration and optimization of Transformer-based approaches in dynamic MRI reconstruction.  

# Flowchart
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/CST-Net.png)  
# Result presentation
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result1.png) <img awidth="500" height="200"/> 
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result2.png) <img awidth="500" height="200"/> 
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result3.png) <img awidth="500" height="150"/> 
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result4.png) <img awidth="500" height="150"/> 

# Datasets
[`dataset download`](https://github.com/yhao-z/T2LR-Net)

# Note
For any questions, feel free to email me at wangb@nim.ac.cn.  
If you find our work useful in your research, please cite our paper ^.^

# Acknowledgments
[`SLR-Net`](https://github.com/Keziwen/SLR-Net),[`L+S-Net`],[`DUS-Net`](https://github.com/yhao-z/DUS-Net),[`T2LR-Net`](https://github.com/yhao-z/T2LR-Net),
