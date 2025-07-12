---
layout: single
title:  "Unified Multimodal Explanations Through Decomposition"
header:
  teaser: "../../images/multimodalblog/cats.png"
categories: 
  - Explainability
tags:
  - nlp
  - explainability
---
*Decomposition allows us to create unified pairwise explanations of a multimodal language model.*

## Introduction
In my [previous post](/explainability/explainability), I described a technique for explainability that involves decomposing the intermediate hidden states of a language model into a set of source and target states. By doing so, we can retrieve the exact contribution of each token to the hidden state of another token, which can then be used to explain the output from any number of downstream tasks. While writing it, I thought about how this can be applied to transformer models that accept inputs of various types (images, audio, etc.). While this does bring various unique challenges, I was finally able to write an implementation of multimodal decomposition that creates cross-modality explanations.

## Why is this important?
Multimodal language models are becoming ubiquitous and all current methods for transformer model explainability are unimodal, with image explanations being considered independently of text explanations. In the paper "Rethinking Explainability in the Era of Multimodal AI", Chirag Agarwal argues that "unimodal explanations systematically misrepresent and fail to capture the cross-modal influence that drives multimodal model decisions, and the community should stop relying on them for interpreting multimodal models." [1]. Becuase of this, there is a strong need to develop model techniques that can create unified explanations for both text and other model inputs.

<img src="../../images/multimodalblog/imgexample.png" width="600"/>

*Caption: A set of 2 images with explanation heatmaps using the XRAI technique [3]. Sections containing a bird are highlighted in red while non-bird sections are black.*

## Multimodal Decomposition Basics
As described previously, decomposition works due to the fact that most operations performed by a transformer model are linear. This means that we can run each model step on the decomposed hidden state while still retaining the ability to collapse it into the undecomposed version. By handling the nonlinear MLP step using linear approximation, we are able to run through the entire model and get a decomposed, fully attributed hidden state from the model.

Since the vision encoders used by multimodal language models are also transformer-based, the same logic can be used to create a decomposed image embedding. After the image embedding is retrieved, a linear transformation is applied to the image embedding to obtain a set of tokens with the embedded image.

## Difficulties
While the overall logic behind multimodal decomposition is basic, the actual implementation of it has a few complications:

### Space Complexity
One obvious difference between a piece of text and an image is that an image is two-dimensional. What this means for decomposition is that a decomposed representation of an image is much larger than any decomposed text embedding. For example, a 64 x 64 grid of image patches is decomposed into 64 x 64 x 64 x 64, or 16,777,216 pairwise attributions. If we use a model with a hidden size of 768 and a dtype of float32, the decomposed tensor would use approximately 51GB of RAM and take an absurdly long time to calculate. For large text inputs, this problem can be mitigated by decomposing a subset of the tokens, since many tasks use only the final token's hidden state. However, this is not possible for images, since we pretty much always want to decompose the entire image.

To solve this, I added a 'scaling factor' to the decomposed embedding that pools square sets of image patches before decomposing. In our 64 x 64 example, a scaling factor of 2 would use a 2x2 pooled grid of image patches, making the decomposed embedding have  a size of 64 x 64 x 32 x 32. This reduces the size of the tensor by a factor of 4, meaning it only takes up 12GB of space. This scaling factor can be dynamically adjusted depending on available memory.

### Integration With Text Embeddings
Because we rescale the size of the images, sizes of the source and target dimensions no longer match directly. For example, if we have 10 text tokens and 64 image tokens with a scaling factor of 2, the full decomposed embedding would be of size (10+64) x (10+32) = 74 x 42. Additionally, this relationship between source and target tokens can be either one-to-many OR many-to-one, since our scaling factor works independently of the image scaling that occurs during regular model inference (i.e. if the regular model scaling reduces to 8 image tokens and our scaling factor reduces to 16).

## Results
I modified the [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) model for decomposition in order to test performance. Since I'm running the model on a laptop with limited memory, I set the scaling factor to 4 to get an 8x8 grid of image attributions along with the full set of text attributions. I used the following image along with the text "Describe this image." as input and retrieved a (truncated) response.

<img src="../../images/multimodalblog/cats.png" width="600"/>

*Caption: The sample input used for multimodal decomposition. It contains two cats lying down on a pink blanket.*

> User: Describe this image.

> Assistant: The image depicts two cats lying on a pink blanket. The cat on the left is smaller and has a lighter color, while the cat on the right is larger and has a darker color. Both cats are lying on their sides, facing opposite directions. The smaller cat is positioned on the left side of the image, while the larger cat is positioned on the right side.\n\nThe background of the image is a pink blanket, which is likely a blanket used for pets. The blanket has a soft

I then ran the entire text through the decomposed model and calculated the delta log-probability for each token as an explanation metric. Specifically, I looked at the token importances to generate the token following "depicts". 

<img src="../../images/multimodalblog/textexplanation.png" width="1000"/>

*Caption: Text explanation for the above sequence of text. Various image tokens are highlighted with different shades of red, with the darkest tokens being the phrase "The image depicts".*

By transforming the image tokens back into image patches, we can visualize the importance of each section of the image:

<img src="../../images/multimodalblog/imgexplanation.png" width="600"/>

*Caption: Explanations for each section of the above image. A few sections are highlighted in green, with most sections being completely transparent.*

Interestingly, we can see that very few patches of the image were important for generating the word "two". In fact, we observe the same pattern for every generated token. In order to get a more complete picture, we can try looking at the average attribution across multiple tokens. If we look at the maximum image attributions for the model to generate the word "cat" or "cats", we can observe whether the model utilizes the image sections that contain a cat:

<img src="../../images/multimodalblog/catexplanation.png" width="600"/>

*Caption: Explanations for each section of the above image. More sections are highlighted in green, most of which are sections containing a cat.*

From the image, we can see that most of the time the model does use sections containing a cat to generate the word 'cat(s)'. However, it's not as clear as other image-only or text-only explainability metrics like XRAI. I also noticed that the earlier image tokens tend to have higher relevance than later ones, potentially showing how a model might use exclusively the first section that matches, rather than utilizing all relevant image sections. This could also interfere with the delta log-probability metric we're using. An individual section containing a cat might be marked as low relevance because ablating that single section causes the model to use all of the other 'cat' sections. This provides an explanation for the apparent noisiness of each explanation.

<img src="../../images/multimodalblog/explanations.png" width="600" />

*Caption: A set of model explanations for the same question using various techniques [2].*

## Conclusion
Decomposition gives us the ability to create explanations for the outputs of multimodal transformer models, which is crucial for understanding the behavior of said models. As far as I know, this is the first technique that directly computes attributions between image and text data. These explanations tend to be fairly noisy when it comes to images, highlighting the need for more robust multimodal explainability techniques. 

In the future, I hope to continue to explore decomposition in both text and image scenarios. To understand the quality decomposition compared to other techniques, I plan on using the [OpenXAI library](https://open-xai.github.io/) and leaderboard. I also hope to apply this technique to larger models, as the quality of the explanation could scale with the quality of the model response itself.

## References

[1]: Agarwal, C. (2025). Rethinking Explainability in the Era of Multimodal AI. arXiv preprint arXiv:2506.13060.

[2]: Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). Attnlrp: attention-aware layer-wise relevance propagation for transformers. arXiv preprint arXiv:2402.05602.

[3]: Munn, M., & Pitman, D. (2022). Explainable AI for Practitioners. Retrieved from https://books.google.com/books?id=-9mYEAAAQBAJ