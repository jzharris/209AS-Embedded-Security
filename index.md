# Combatting the Vulnerabilities of Split Learning

### Authors
* Zach Harris: UCLA, M.S. in ECE, _jzharris@g.ucla.edu_
* Hamza Khan: UCLA, M.S. in ECE, _hamzakhan@g.ucla.edu_

### Abstract
As big data analytics becomes rooted in critical fields such as health, finance, economics, and politics,
the privacy of the data used to train the models must remain upheld. For example, health information from patients is confidential
and must abide by patient confidentiality agreements. This information can still be used to collaboratively train a 
deep learning model while maintaining privacy through methods such as Federated Learning. 
Federated Learning, however, has been shown to expose backdoors which make the training data vulnerable
to unwanted and undetected data retrieval or model poisoning attacks during training. A recent method called Split 
Learning claims to provide a secure way of collaboratively training deep learning models. The vulnerabilities of this
method have not been fully investigated, however. In this work, we first present vulnerabilities of Federated Learning 
involving raw data retrieval and input data poisoning. We focus our investigation to label flipping and backdoor attacks 
using methods presented from [[1]](#1). Secondly, we 
investigate how such attacks can be used to exploit vulnerabilities in Split Learning. Lastly, we present methods to
prevent or mitigate the attacks in order to increase the resilience of the Split Learning method against adversaries.

### Citations

<ol>
##### Vulnerabilities of Federated Learning

    <li id="1">
    Zhang, Jiale, et al. "Poisoning Attack in Federated Learning using Generative Adversarial Nets." 2019 18th IEEE International Conference On Trust, Security And Privacy In Computing And Communications/13th IEEE International Conference On Big Data Science And Engineering (TrustCom/BigDataSE). IEEE, 2019.
    </li>

2. Kairouz, Peter, et al. "Advances and open problems in federated learning." arXiv preprint arXiv:1912.04977 (2019).

3. Bhagoji, Arjun Nitin, et al. "Analyzing federated learning through an adversarial lens." arXiv preprint arXiv:1811.12470 (2018).

##### Label Flipping Attacks

1. Biggio, Battista, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines." arXiv preprint arXiv:1206.6389 (2012).
2. Huang, Ling, et al. "Adversarial machine learning." Proceedings of the 4th ACM workshop on Security and artificial intelligence. 2011.

##### Backdoor Attacks

1. Chen, Xinyun, et al. "Targeted backdoor attacks on deep learning systems using data poisoning." arXiv preprint arXiv:1712.05526 (2017).
2. Bagdasaryan, Eugene, et al. "How to backdoor federated learning." arXiv preprint arXiv:1807.00459 (2018).

##### Split Learning Method

1. Vepakomma, Praneeth, et al. "Split learning for health: Distributed deep learning without sharing raw patient data." arXiv preprint arXiv:1812.00564 (2018).
2. Gupta, Otkrist, and Ramesh Raskar. "Distributed learning of deep neural network over multiple agents." Journal of Network and Computer Applications 116 (2018): 1-8.
3. Vepakomma, Praneeth, et al. "No Peek: A Survey of private distributed deep learning." arXiv preprint arXiv:1812.03288 (2018).
4. Vepakomma, Praneeth, et al. "Reducing leakage in distributed deep learning for sensitive health data." arXiv preprint arXiv:1812.00564 (2019).
5. Singh, Abhishek, et al. "Detailed comparison of communication efficiency of split learning and federated learning." arXiv preprint arXiv:1909.09145 (2019).
6. Sharma, Vivek, et al. "ExpertMatcher: Automating ML Model Selection for Users in Resource Constrained Countries." arXiv preprint arXiv:1910.02312 (2019).
7. Sharma, Vivek, et al. "ExpertMatcher: Automating ML Model Selection for Clients using Hidden Representations." arXiv preprint arXiv:1910.03731 (2019).
8. Poirot, Maarten G., et al. "Split Learning for collaborative deep learning in healthcare." arXiv preprint arXiv:1912.12115 (2019).

##### Attacker Detection and Mitigation

1. Chen, Huili, et al. "Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks." Proceedings of the 28th International Joint Conference on Artificial Intelligence. AAAI Press. 2019.
2. Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "Mitigating sybils in federated learning poisoning." arXiv preprint arXiv:1808.04866 (2018).


</ol>