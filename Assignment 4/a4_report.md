| hidden act_func | L2_norm | dropout | accuracy |
| :-------------: | :-----: | :-----: | :------: |
| ReLU | true    | true    | 76.86 |
| ReLU | true    | false    | 78.33 |
| ReLU | false    | true    | 77.38 |
| ReLU | false | false    | 75.22 |
| sigmoid | true    | true    | 74.14 |
| sigmoid | true    | false    | 50.00 |
| sigmoid | false    | true    | 79.44 |
| sigmoid | false | false    | 75.52 |
| tanh | true    | true    | 74.62 |
| tanh | true    | false    | 77.58 |
| tanh | false    | true    | 79.75 |
| tanh | false | false    | 76.59 |

- __What effect do activation functions have on your results?__

  These three activation functions perform similarly, while ___tanh___ seems the best for this data.  

  I think that _tanh_ is suitable for classifying problem because its function is $tanh(x) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ , which means approximating a classifier function as combinations of tanh is easier than _ReLu_. 

  And actually, _sigmoid_ also works well for a classifier but it has gradient vanishing and zero-centered problems.

  For _ReLU_, it leads to faster training process and convergence without gradient vanishing, which is shown in my results below.

- __What effect does addition of _L2-norm_ regularization have on the results? __

  With _L2-norm_, the results aren't improved, and even __worse__ sometimes. Since _L2-norm_ penalties in some sense discourage sparsity by yielding diminishing returns as elements are moved closer to 0, while our data is sparse.  This worsen _sigmoid_ 's zero-centered problem.

- __What effect does dropout have on the results?__

  With _dropout_, the model performs __better__.   _dropout_ prevent over-fitting, reducing interdependent learning amongst the neurons. It helps three action functions, especially _sigmoid_, perform better. There isn't a rule for dropout rate, it is found that 0.5 as dropout rate performs best for my data.

  







![a4_result](./a4_result.png)