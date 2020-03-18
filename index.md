# Combatting the Vulnerabilities of Split Learning

### Authors
* Zach Harris: UCLA, M.S. in ECE, _jzharris@g.ucla.edu_
* Hamza Khan: UCLA, M.S. in ECE, _hamzakhan@g.ucla.edu_

###### subsubsection

### Abstract
As big data analytics becomes rooted in critical fields such as health, finance, economics, and politics,
the privacy and integrity of the training data must be upheld. For example, health information from patients is confidential
and must abide by patient confidentiality agreements. This information can still be used to collaboratively train a 
deep learning model while maintaining privacy through methods such as Federated Learning. 
Federated Learning, however, has been shown to expose backdoors which make the data liable
to undetectable data retrieval and model poisoning attacks during training. A recent method called Split 
Learning claims to provide a secure way of collaboratively training deep learning models. The vulnerabilities of this
method have not been fully investigated, however. In this work, we first present vulnerabilities of Federated Learning 
involving raw data retrieval and input data poisoning [[1-2]](#1), focussing on 
label flipping attacks [[3-5]](#3) and backdoor attacks [[6-7]](#6). 
Secondly, we introduce Split Learning [[8-15]](#8) and investigate its susceptibility to label flipping and backdoor attacks. 
Lastly, we present methods to prevent or mitigate attacks on Split Learning inspired from previous mitigation methods [[16-17]](#16).

### Project Goals and Specific Aims

1. Simulate Split Learning
2. Present the vulernabilities of Split Learning
    - Label poisoning attacks [[3-5]](#3)
    - Backdoor attacks [[6-7]](#6)
3. Harden Split Learning
    - Black-box trojan detection [[16]](#16)
    - Mitigating federated learning poinsoning [[17]](#17)

### Experiments

1. Show that we can approximate the Black-box model
    - Black-box complexity
    - Black-box accuracy
    
2. Optimize the Black-box model approximation
    - Approximate Black-box by recognizing gradient patterns during training

3. Adapt GAN label poisoning attack from [[3]](#3) to target Split Learning
    - Use gradient techniques to improve GAN attack performance

4. Perform backdoor label poisoning attacks [[6-7]](#6)
5. Evaluate Split Learning attack detection and prevention methods [[16-17]](#16)

### TEST

![](Eval_0_clean.gif)

### Citations

##### Vulnerabilities of Federated Learning

<ol>
    <li id="1">
    Kairouz, Peter, et al. "Advances and open problems in federated learning." arXiv preprint arXiv:1912.04977 (2019).
    </li>
    <li id="2">
    Bhagoji, Arjun Nitin, et al. "Analyzing federated learning through an adversarial lens." arXiv preprint arXiv:1811.12470 (2018).
    </li>
</ol>

##### Label Flipping Attacks

<ol start="3">
    <li id="3">
    Zhang, Jiale, et al. "Poisoning Attack in Federated Learning using Generative Adversarial Nets." 2019 18th IEEE International Conference On Trust, Security And Privacy In Computing And Communications/13th IEEE International Conference On Big Data Science And Engineering (TrustCom/BigDataSE). IEEE, 2019.
    </li>
    <li id="4">
    Biggio, Battista, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines." arXiv preprint arXiv:1206.6389 (2012).
    </li>
    <li id="5">
    Huang, Ling, et al. "Adversarial machine learning." Proceedings of the 4th ACM workshop on Security and artificial intelligence. 2011.
    </li>
</ol>

##### Backdoor Attacks

<ol start="6">
    <li id="6">
    Chen, Xinyun, et al. "Targeted backdoor attacks on deep learning systems using data poisoning." arXiv preprint arXiv:1712.05526 (2017).
    </li>
    <li id="7">
    Bagdasaryan, Eugene, et al. "How to backdoor federated learning." arXiv preprint arXiv:1807.00459 (2018).
    </li>
</ol>

##### Split Learning Method

<ol start="8">
    <li id="8">
    Vepakomma, Praneeth, et al. "Split learning for health: Distributed deep learning without sharing raw patient data." arXiv preprint arXiv:1812.00564 (2018).
    </li>
    <li id="9">
    Gupta, Otkrist, and Ramesh Raskar. "Distributed learning of deep neural network over multiple agents." Journal of Network and Computer Applications 116 (2018): 1-8.
    </li>
    <li id="10">
    Vepakomma, Praneeth, et al. "No Peek: A Survey of private distributed deep learning." arXiv preprint arXiv:1812.03288 (2018).
    </li>
    <li id="11">
    Vepakomma, Praneeth, et al. "Reducing leakage in distributed deep learning for sensitive health data." arXiv preprint arXiv:1812.00564 (2019).
    </li>
    <li id="12">
    Singh, Abhishek, et al. "Detailed comparison of communication efficiency of split learning and federated learning." arXiv preprint arXiv:1909.09145 (2019).
    </li>
    <li id="13">
    Sharma, Vivek, et al. "ExpertMatcher: Automating ML Model Selection for Users in Resource Constrained Countries." arXiv preprint arXiv:1910.02312 (2019).
    </li>
    <li id="14">
    Sharma, Vivek, et al. "ExpertMatcher: Automating ML Model Selection for Clients using Hidden Representations." arXiv preprint arXiv:1910.03731 (2019).
    </li>
    <li id="15">
    Poirot, Maarten G., et al. "Split Learning for collaborative deep learning in healthcare." arXiv preprint arXiv:1912.12115 (2019).
    </li>
</ol>

##### Attacker Detection and Mitigation

<ol start="16">
    <li id="16">
    Chen, Huili, et al. "Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks." Proceedings of the 28th International Joint Conference on Artificial Intelligence. AAAI Press. 2019.
    </li>
    <li id="17">
    Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "Mitigating sybils in federated learning poisoning." arXiv preprint arXiv:1808.04866 (2018).
    </li>
</ol>
