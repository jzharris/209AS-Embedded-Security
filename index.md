# Combatting the Vulnerabilities of Split Learning

<!-- # Header 1
## Header 2
### Header 3
#### Header 4
##### Header 5
###### Header 6 -->

### Authors
* Zach Harris: UCLA, M.S. in ECE, _jzharris@g.ucla.edu_
* Hamza Khan: UCLA, M.S. in ECE, _hamzakhan@g.ucla.edu_

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
==Lastly, we present methods to prevent or mitigate attacks on Split Learning inspired from previous mitigation methods [[16-17]](#16).==

## I. Introduction

## II. Background

### A. Standard Approaches to Distributed Deep Learning

##### Federated Learning

##### Large Batch Synchronous SGD


### B. Split Learning

##### Methods

##### Security Benefits


### Generative Adversarial Networks (GANs)

<!-- ## Overall Project Goals -->
## III. Motivation

### A. What we are doing

Investigating data privacy and model poisoning vulnerabilities for systems using the Split Learning paradigm to train a shared model.

### B. Why it is important

To show that two claims [[9]](#9) made by Split Learning are invalid:
1. Keeps training data of clients private
2. Mitigates malicious attacks by making a portion of the shared model off-limits to clients

### C. How it is done today

No previous work has shown the vulnerabilities of Split Learning. We extend present state-of-the-art methods to attack Split Learning.

<!-- 1. Simulate Split Learning
2. Present the vulernabilities of Split Learning
    - Label poisoning attacks [[3-5]](#3)
    - Backdoor attacks [[6-7]](#6)
3. Harden Split Learning
    - Black-box trojan detection [[16]](#16)
    - Mitigating federated learning poinsoning [[17]](#17) -->

### D. Specific Aims

1. Gather statistics for performing label poisoning and black-box attacks
2. Simulate the Split Learning method
3. Approximate Split Learning’s black-box model
4. Evaluate vulnerability of Split Learning to a label poisoning attack
5. Evaluate vulnerability of Split Learning to a data estimation attack


<!-- ## Technical Approach -->
## IV. Methods

### A. Label Poisoning Attack

### B. GAN Poisoning Attack

### C. GAN System Verification

### D. Black-box Attack


## V. Implementation

### A. Split Learning simulation

### B. Attack setup


## VI. Experiments

### A. Label Poisoning

##### Success metrics

### B. GAN System Verification

##### Success metrics

### C. GAN Black-box Attack

##### Success metrics

### D. Label Poisoning using GAN Images

##### Success metrics

<!-- 1. Show that we can approximate the Black-box model
    - Black-box complexity
    - Black-box accuracy
    
2. Optimize the Black-box model approximation
    - Approximate Black-box by recognizing gradient patterns during training

3. Adapt GAN label poisoning attack from [[3]](#3) to target Split Learning
    - Use gradient techniques to improve GAN attack performance

4. Perform backdoor label poisoning attacks [[6-7]](#6)
5. Evaluate Split Learning attack detection and prevention methods [[16-17]](#16) -->

<!-- ### TEST

![](Eval_0_clean.gif) -->

## VII. Results

### A. Label Poisoning

### B. GAN System Verification

### C. GAN Black-box Attack

### D. Label Poisoning using GAN Images

## VIII. Related Work

<!-- TODO: add Black-box attack papers -->

## IX. Conclusion

## X. Citations

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

##### Black-box model attacks
<ol start="18">
    <li id="18">
    https://github.com/MadryLab/mnist_challenge#black-box-leaderboard-original-challenge
    </li>
    <li id="19">
    Ilyas, Andrew, et al. "Black-box adversarial attacks with limited queries and information." arXiv preprint arXiv:1804.08598 (2018).
    </li>
    <li id="20">
    Xiao, Chaowei, et al. "Generating adversarial examples with adversarial networks." arXiv preprint arXiv:1801.02610 (2018).
    </li>
</ol>

##### PGD-based black-box attacks
<ol start="21">
    <li id="21">
    Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
    </li>
    <li id="22">
    Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).
    </li>
    <li id="23">
    Papernot, Nicolas, et al. "Practical black-box attacks against machine learning." Proceedings of the 2017 ACM on Asia conference on computer and communications security. 2017.
    </li>
</ol>