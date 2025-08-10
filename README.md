<h1 align="center">IMU: Influence-guided Machine Unlearning</h1>
This is the official code repository for the paper <a href="https://arxiv.org/abs/2508.01620">IMU: Influence-guided Machine Unlearning</a>.
<h2 align="left">Abstract</h2>
Recent studies have shown that deep learning models are vulnerable to attacks and tend to memorize training data points, raising significant concerns about privacy leakage. This motivates the development of machine unlearning (MU), i.e., a paradigm that enables models to selectively forget specific data points upon request. However, most existing MU algorithms require partial or full fine-tuning on the retain set. This necessitates continued access to the original training data, which is often impractical due to privacy concerns and storage constraints. A few retain-data-free MU methods have been proposed, but some rely on access to auxiliary data and precomputed statistics of the retain set, while others scale poorly when forgetting larger portions of data. In this paper, we propose Influence-guided Machine Unlearning (IMU), a simple yet effective method that conducts MU using only the forget set. Specifically, IMU employs gradient ascent and innovatively introduces dynamic allocation of unlearning intensities across different data points based on their influences. This adaptive strategy significantly enhances unlearning effectiveness while maintaining model utility. Results across vision and language tasks demonstrate that IMU consistently outperforms existing retain-data-free MU methods.
<h2 align="left">Getting Started</h2>
IMU can be applied to different tasks such as image classification, person Re-ID, sequence modeling and large language model unlearning. You can click the link below to access a more detailed guide.<br>
 <p><a href="https://github.com/goodluckisallyouneed/IMU/tree/main/Classification#readme">
IMU for image classification and person Re-ID
</a></p>

<p><a href="https://github.com/goodluckisallyouneed/IMU/blob/main/synthetic/README.md">
IMU for sequence modeling and large language model unlearning
</a></p>

<h2 align="left">Citing this work</h2>
<pre>
@misc{fan2025imuinfluenceguidedmachineunlearning,
      title={IMU: Influence-guided Machine Unlearning}, 
      author={Xindi Fan and Jing Wu and Mingyi Zhou and Pengwei Liang and Dinh Phung},
      year={2025},
      eprint={2508.01620},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.01620}, 
}
</pre>
