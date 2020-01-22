# Anomaly detection for Split Learning

### Authors
* Zach Harris: UCLA, M.S. in ECE, _jzharris@g.ucla.edu_
* Hamza Khan: UCLA, M.S. in ECE, _hamzakhan@g.ucla.edu_

### Abstract
As Big Data Analytics becomes engrained in important fields such as health, finance, economics, and politics,
the privacy of the data and models must be investigated. For example, health information from patients is confidential
and must abide by patient confidentiality agreements. Previous methods to regulate raw data sharing during the training 
process include Federated Learning. This method, however, contains back-doors which makes the training data vulnerable
to unwanted and undetected data retrieval and model poisoning attacks during training. A recent method called Split 
Learning supposedly provides a secure way of collaboratively training deep learning models. The vulnerabilities of this
method have not been fully investigated, however. In this work, we provide a thorough investigation as following.
First, we present the vulnerabilities of data retrieval and model poisoning of Federated Learning. We show how these
attacks can be likewise adapted to Split Learning. Secondly, we present the Split Learning method and attempt to
exploit its weaknesses in order to retrieve private raw data and poison the trained model. Lastly, we present a means to
counter the attacks we present and make Split Learning more resilient to such attacks.