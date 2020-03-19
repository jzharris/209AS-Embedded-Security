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
label poisoning attacks [[3-5]](#3) and backdoor attacks [[6-7]](#6). 
Secondly, introduce Split Learning [[8-15]](#8) and investigate its susceptibility to label poisoning and backdoor attacks. Finally, we introduce an attack pipeline consiting of a Generative Adversarial Network (GAN) and apply black-box attack techniques to the pipeline to improve upon the attack's success.

## I. Introduction

<!-- TODO: give an overarching view of Split Learning, and some examples for what ways Split Learning is secure. Mention Federated Learning and draw connections between FL and SL. -->

## II. Background

### A. Standard Approaches to Distributed Deep Learning

##### Federated Learning

##### Large Batch Synchronous SGD


### B. Split Learning

##### Methods

##### Security Benefits


### C. Generative Adversarial Networks (GANs)

<!-- ## Overall Project Goals -->
## III. Motivation

### A. What we are doing

Investigating data privacy and model poisoning vulnerabilities for systems using the Split Learning paradigm to train a shared model.

We are investigating the data privacy and model poisoning vulnerabilities for systems using the Split Learning paradigm to collaboratively train models. We are mounting a label poisoning attack on a class of the model that is kept private to us. For example, let us assume that a Split Learning model is being trained to predict MNIST digits and an attacker is training the model to increase the accuracy of the model on the '0' class. Let us assume that the attacker seeks to poison the '1' class of the model. We speculate three reasons why it is plausible for an attacker to target a particular class of the model, when they don't know the data behind that class:

1. They have deduced that one of the clients they wish to attack may have the '1' class. In this case, the attacker does not know what data is present in the '1' class, however they wish to reduce the accuracy of the portion of the model that their targetted client is training.
2. Instead of focussing on a particular client, the attacker is trying to optimize the potency of their attack on the model over all classes. For example, perhaps the attacker realizes that there is an increased potency in their attack when they mislabel images from the '1' class as being from the '7' class, but a decreased potency when they mislabel images from the '8' class as the '7' class. There is therefore a better chance for their attack to work by choosing to poison the '1' class over the '8' class.
3. The attacker could have chosen this class arbitrarily. They could be trying to negatively impact the accuracy of any of the victim classes while preserving the accuracy of their own class.

### B. Why it is important

Split Learning has made specific claims [[9]](#9) that gives the training pipeline an extra edge in security over other collaborative training methods such as Federated Learning.

Their first claim is that the training data of disjoint clients cannot be accessed by one another, and a client's dataset is garaunteed to be secure regardless of the number of malicious clients present. This claim stems from the fact that the client's dataset never leaves their local system: the only information shared by the client's machine are the activations of the last client-side layer. The client-side layers introduce noise and perform nonlinear operations on the original input data. The information sent to the server is therefore not the client's raw data, but can be thought of as a cipher calculated by the client's private layers. [[TODO]]() The cipher is weak because these client-side layers are shared amongst clients, which means that a maliscious client can recover the private data if they intercepted the victim client's transmission. However, since we are dealing with client-server communication, the transmission can be encrypted using public key encryption [[TODO]](). However, an attacker outside of the Split Learning process would not be able to derive the client's dataset regardless of extra encyption. Split Learning therefore easily mitigates snooping attacks and keeps all of the client's data private.

Their second claim is that malicious client attacks are mitigated since a portion of the shared model is off-limits to all clients. Zhang, Jiale, et al. [[3]](#3) showed that the private datasets of clients can be approximated when all clients have access to the model while it is training. Their work involved using a GAN to implement a label poisoning attack where the GAN's Discriminator model was replaced with the public model. By doing so, the GAN's generator was able to generate a convincing approximation of all classes that the attacker did not have access to. In order to mitigate this vulnerability, Split Learning keeps a portion of the model off-limits to the clients. This way, it is incredibly difficult to mount an attack using a GAN like before, because most of the model is now unkown to the attacker.

### C. How it is done today

No previous work has shown the vulnerabilities of Split Learning. Gupta et al. [[9]](#9) mention their future work will be to investigate vulnerable aspects to Split Learning. We have performed this work in order to help in the investigation. Because no prior work has mounted an attack on Split Learning, we extend present state-of-the-art methods to perform an attack. First, we simulate the Split Learning pipeline from [[9]](#9). Second, we incorporate the GAN attack from [[3]](#3). Finally, we improve our new pipeline with black-box attacks such as FGSM [[21-23]](#21) and advGAN [[20]](#20). We decided to test the aspects of these two methods in particular by comparing the performance of different black-box attack attempts on the MNIST dataset in the MIT _MNIST Adversarial Examples Challenge_ [[18]](#18).

### D. Specific Aims

In this work, we aim to perform the following five tasks:

1. Gather statistics for performing label poisoning and black-box attacks 
    - In order to ensure our attack will work once we 
3. Simulate the Split Learning method
4. Approximate Split Learning’s black-box model
5. Evaluate vulnerability of Split Learning to a label poisoning attack
6. Evaluate vulnerability of Split Learning to a data estimation attack


<!-- ## Technical Approach -->
## IV. Methods

### A. Label Poisoning Attack

### B. GAN Poisoning Attack

In order to perform a label poisoning attack on a real Split Learning system, we require knowledge of the other clients' private datasets. For the following example, consider that the attackers represent a portion of the '0' class and desire to poison the clients who are training the '1' class. The attacker could be doing this for any of the reasons we presented in the Motivation section.

##### 8-step process

![](report/8steps.gif)

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

## VII. Results

### A. Label Poisoning

### B. GAN System Verification

### C. GAN Black-box Attack

### D. Label Poisoning using GAN Images

## VIII. Related Work

<!-- TODO: add Black-box attack papers -->

## IX. Futer Work

<!-- ==Lastly, we present methods to prevent or mitigate attacks on Split Learning inspired from previous mitigation methods [[16-17]](#16).== -->

<!-- ## X. Conclusion -->

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

##### Label Poisoning Attacks

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