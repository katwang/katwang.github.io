---
layout: default
title:  "Sleep Habits & Mental Health"
date:   2016-07-21 12:00:00 -0400
categories: jekyll update
---

 # Exploration of the Relationship Between Self-Reported Sleep Habits and Mental Health

 **Dataset**: [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs)

[test](https://github.com/katwang/BST234Project/blob/master/BST234_SKAT.ipynb)

test2 - 

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "BST234 - SKAT.ipynb",
      "version": "0.3.2",
      "views": {},
      "default_view": {},
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "metadata": {
        "id": "uzXRY29Amg-R",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Due: Friday, 27th\n",
        "* [Presentation](https://docs.google.com/presentation/d/14f3YDwOwMksdrS1iqbGA0bVJOeJOH59DFEgF1LtpQhU/edit?usp=sharing)\n",
        "* Paper\n",
        "* Code\n",
        "\n",
        "\n",
        "### Intro\n",
        "\n",
        "\n",
        "Single Nucleotide Polymorphisms (SNPs) are variations in the genome which result in sinlge nucleotide differnces. They occur once every 300 nucleotides on average, and  there are roughly 10 million SNPs in the human genome. As Genome Wide Association Studies (GWAS) studies have steadily accumulated data, robust means of determining genetic variants is paramount. Simultaneously, computing parallelization and speed has accelerated machine learning frameworks. Here we implement sequence kernel association test (SKAT) - a powerful test for rare varaint associations that builds upon kernel machine regression. \n",
        "\n",
        "\n"
      ]
    },
    {
      "metadata": {
        "id": "zMdnHsDUnxho",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### SKAT Model\n",
        "\n",
        "\n",
        "Under the assumptions presented by Wu in 2011, we fit the model \n",
        "\n",
        "$$\n",
        "logit \\space P(y_i=1)=\\alpha_0+\\alpha'X_i+\\beta'G_i\n",
        "$$\n",
        "\n",
        "where $y_i$ is the phenotype of interest, $X$ is a matrix of covariates, and $G$ is an $mxn$ genotype matrix, where $m$ is the number of subjects and $n$ is the number of loci. \n",
        "\n",
        "In our project, we're looking at the association of affection status (cases vs. controls) and 50 loci from a study of 20,000 participants. This gives us the matrix $G_{20000\\times 50}$.\n",
        "\n",
        "### Test of Significance\n",
        "\n",
        "With SKAT, we test the null hypothesis of $H_0:\\beta=0$ (all genotypes have no association) with $H_1:\\beta \\neq 0$ (the effects of the loci follow a Gaussian distribution) using the test statistic $Q$ defined as the following:\n",
        "\n",
        "$$\n",
        "Q=(y-\\hat{\\mu})'K(y-\\hat{\\mu})\n",
        "$$\n",
        "\n",
        "where \n",
        "\n",
        "$K=GWG'$, $W=diag(w_1,\\dots, w_n)$ contains the weights of the n variants, and $\\hat{\\mu}$ is the predicted mean of y under the null hypothesis, that is $\\hat{\\mu} = logit^{-1}(\\hat{\\alpha_0} + \\hat{\\alpha}'X_i)$\n",
        "\n",
        "$Q$ can also be represented by \n",
        "\n",
        "$$\n",
        "Q = \\sum_{j=1}^m w_j \\frac{{1}}{{2\\hat{\\sigma^2}}} \\left(\\sum_{i=1}^n Y_{i}(Y_i - \\hat{\\mu_i})G_{m,n}\\right)^2\n",
        "$$\n",
        "\n",
        "where $\\hat{\\sigma^2}$ is the sample variance of $Y - \\hat{\\mu}$ and the weights are denoted $w_j$. Weights were chosen such that $E(w_j) = MAF_j$. \n",
        "\n",
        "Under the null hypothesis of no genetic associations, $Q$ follows a sum of $\\chi^2$ variables:\n",
        "\n",
        "$$\n",
        "Q \\sim \\sum_{i=Y_i}^n{\\lambda_i \\chi^2_{1,i}}\n",
        "$$\n",
        "\n",
        "where $\\lambda_1,...,\\lambda_n$ are the eigenvalues of $P_{0}^{0.5}KP_{0}^{0.5}$ where $P_{0} = V - V\\tilde{X}\\left(\\tilde{X}'V\\tilde{X}\\right)^{-1}\\tilde{X}'V$ where\n",
        "$V = \\hat{\\sigma}^2I$ \n",
        "\n",
        "The definition of $\\lambda_1,...,\\lambda_n$ as defined above is the exact calculation as proposed by the original authors of SKAT (Wu 2011). Lumley proposed that the eigenvalues be derived from $H$ such that\n",
        "\n",
        "$\\lambda_1,...,\\lambda_n$ are the eigenvalues of $H=\\tilde G \\tilde G^T$, where $\\tilde G = [(w\\odot G)(I_n-X(X^TX)^{-1}X^T)]/\\sqrt{2}$\n",
        "\n",
        "Calculating $H$, an $n\\times n$ matrix, has a complexity of $O(n^3)$. If $m>>n$, we should use the smaller matrix $H'=\\tilde G^T \\tilde G$. Regardless of how matrix H is formulated, the computational complexity of calculating all the eigenvalues of a n x n will be $O(n^3)$.  \n",
        "\n",
        "\n",
        "### Computing the P-Values\n",
        "\n",
        "Generally we can approximate the scaled $\\chi^2$ distribution of $Q\\sim a\\chi^2_v$ using the Satterthwaite approach:\n",
        "\n",
        "where we use all $n$ eigenvalues to find the scaling factor $a=\\frac{\\sum_n{\\lambda^2_i}}{\\sum_n{\\lambda_i}}$ and the degrees of freedom $v=\\frac{(\\sum_n{\\lambda_i})^2}{\\sum_n{\\lambda^2}}$.\n",
        "\n",
        "The Satterthwaite approach attempts to approximate the true distribution of Q with a single $a\\chi^2_v$ distribution with scaling factor $a$ and degrees of freedom $v$. Since the Satterthwaite approximation tends to be anticonversative, we employ the method proposed by Lumley in FastSKAT. \n",
        "\n",
        "The adjusted Saatterthwaite approach uses the first $k$ eigenvalues of $H$ to get the following distribution:\n",
        "\n",
        "$$Q \\sim \\left(\\sum_{i=1}^k \\lambda_i \\chi_1^2\\right) +a_k \\chi_{v_k}^2$$\n",
        "\n",
        "where, \n",
        "\n",
        "$$a_k = \\big( \\sum_{i=k+1}^m \\lambda_i^2 \\big)\\big/\\big( \\sum_{i=k+1}^m \\lambda_i \\big)$$\n",
        "\n",
        "$$v_k = \\big( \\sum_{i=k+1}^m \\lambda_i \\big)^2\\big/\\big( \\sum_{i=k+1}^m \\lambda ^2 \\big)$$\n",
        "\n",
        "$$\\sum \\lambda = \\sum_i H_{ii} \\textrm{ and } \\sum_{i,j} \\lambda = \\sum_i H_{ij}^2$$\n",
        "\n"
      ]
    },
    {
      "metadata": {
        "id": "x_B8wjnHESbA",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Discussion\n",
        "\n",
        "We implemented SKAT as a score test following the methodology outlined by Lumley. This iteration of SKAT is presented as a function `fastSKAT`, that takes in the genotype matrix $G$ and returns the test statistic $Q$, matrix $H$, and the eigenvalues of matrix $H$. For this implementation of SKAT, we always assumed the model $P(y_i=1)=\\alpha_0$ under $H_o:\\beta_j = 0$. Specifically, we were interested in the unadjusted associations between genetic variants and phenotype. \n",
        "\n",
        "The `fastSKAT_Pvalue` function then takes the output of `fastSKAT` and returns the p-value calculated using the standard Satterwaite approach as well as the adjusted Satterwaite approximation.\n",
        "\n",
        "\n",
        "### Alternate Options\n",
        "\n",
        "We usually use SKAT when there are many loci and fewer subjects, which may make the calculation of the test statistic $Q$ very computationally intensive. This is because we would end up with a very large $H$ matrix from which we'd have to calculate our eigenvalues. In this case, we could calculate the $k$ largest eigenvalues using a random projection, similar to what we did to estimate the p-value. \n",
        "\n",
        "Using this method, we could approximate $H$ and $\\tilde G$:\n",
        "\n",
        "* $H\\approx$ the $k$ largest eigenvalues of $QHQ^T$\n",
        "* $\\tilde G\\approx$ the singular values of $Q\\tilde G$\n",
        "\n",
        "where $Q$ is derived from the QR decomposition of $(\\Omega H)^T$ given the linear transformation matrix $\\Omega$.\n",
        "\n",
        "The specific data matrix we were given does not have this issue since we only had 50 loci and we ended up with a 50$\\times$ 50 H matrix "
      ]
    },
    {
      "metadata": {
        "id": "KCBPngnAEZD6",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#### Result Interpretation\n",
        "\n",
        "As we can see the result above, the basic Satterthwaite approach tends to be anticonservative, according to the author of fastSKAT, the basic Satterthwaite is only accucrate if it give a p-values larger than $10^{-3}$ in a genome-wide scan. Therefore, in our case, we implemented the improved approach suggested by the fastSKAT. First we use scree plot to decide the number of eigenvalues we should include. We decided to use $k=10$ since that would allows us inclued the most signifcant eigenvalues. Then we calulate the improve p-value, which is 0.032, still smaller than 0.05. Thus, we conclude we have sufficient evidence to claim there is association between the loci and the disease. "
      ]
    },
    {
      "metadata": {
        "id": "APlutvnHq87E",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "--------------------------\n",
        "\n",
        "$$\\mathcal{Q}=(y-\\hat \\mu)'K(y-\\hat \\mu)$$\n",
        "Where, $K=GWG'$, $W=diag(w_1,\\dots, w_p)$ contains the weights of the p variants.\n",
        "\n",
        "$\\hat \\mu$ is the predicted mean of y under $H_0$, \n",
        "\n",
        "$\\hat \\mu =\\hat{a_0}+ X \\hat{\\alpha} = \\hat \\alpha_0$\n",
        "\n",
        "So we have: \n",
        "$$\\mathcal{Q}=y'Ky$$\n",
        "\n",
        "Now, we need to have a choices of $W$(weights)\n",
        "\n",
        "If $w_j$ is close to zero, then the j-th variant makes only a small contribution to Q. Thus, decreasing the weight of noncausal variants and increasing the weight of causal variants can yield improved power. \n",
        "\n",
        "Because in practice we do not know which variants are causal, we propose to set $w_j = \\beta (MAF_j;a_1,a_2)$\n",
        "\n",
        "This is how elini generated data, not sure we can use this to get the $W$\n",
        "\n",
        "{r}\n",
        "mafs<-0.00005+runif(50)*(0.001-0.00005) # maf between 0.00005 and 0.001\n"
      ]
    },
    {
      "metadata": {
        "id": "_cWqMHvVCtw_",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "### Fast eigenvalue calculations (from paper)\n",
        "\n",
        "$\\lambda_j$ are eigenvalues of $H=GG^T$; details on how to calculate this in appendix\n",
        "\n",
        "$\\Omega$: linear transformation matrix\n",
        "\n",
        "$H$: QR Decomp of $(\\Omega H)^T \\to Q\\to$ eigenvalue decomposition of $QHQ^T\\to$ k largest eigenvalue = good approx of $H$\n",
        "\n",
        "$G$: calculate $Q$ from $\\Omega G \\to SVD(QG)\\to$ k largest singular values = good approx of $G$\n",
        "\n",
        "\n",
        "The test statistics Q follows a scaled $\\chi_2$ distribution using a satterwaite approach, and we use a improved version of satterthwaite, and we have: \n",
        "\n",
        "$$ Q \\sim (\\sum_{i=1}^k \\lambda_i \\chi_1^2) +a_k \\chi_{v_k}^2$$\n",
        "\n",
        "Where, \n",
        "\n",
        "$$a_k = \\big( \\sum_{i=k+1}^m \\lambda_i^2 \\big)\\big/\\big( \\sum_{i=k+1}^m \\lambda_i \\big)$$\n",
        "\n",
        "$$v_k = \\big( \\sum_{i=k+1}^m \\lambda_i \\big)^2\\big/\\big( \\sum_{i=k+1}^m \\lambda ^2 \\big)$$\n",
        "\n",
        "$$\\sum \\lambda = \\sum_i H_{ii} \\textrm{ and } \\sum_{i,j} \\lambda = \\sum_i H_{ij}^2$$\n"
      ]
    },
    {
      "metadata": {
        "id": "pvPQpZfPsRWa",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#### References\n",
        "\n",
        "##### The orginal SKAT paper\n",
        "* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135811/\n",
        "\n",
        "\n",
        "##### More info on SKAT\n",
        "* http://gero.usc.edu/CBPH/files/4_9_2013_PAA/15_Kardia_SequenceKernelAssociationTest.pdf\n",
        "\n",
        "* https://www.biorxiv.org/content/biorxiv/early/2017/06/03/140889.full.pdf\n",
        "\n",
        "* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4761292/\n",
        "\n",
        "* https://www.biorxiv.org/content/biorxiv/early/2018/01/29/085639.full.pdf\n",
        "\n",
        "##### Some Python SKAT code\n",
        "\n",
        "* https://github.com/cozygene/RL-SKAT/blob/master/rl_skat.py\n",
        "\n",
        "\n",
        "##### SKAT paper with same documentation as lecture slides\n",
        "\n",
        "* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3440237/pdf/kxs014.pdf"
      ]
    },
    {
      "metadata": {
        "id": "YT6WMzPssPuu",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          }
        }
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Ru_e0fUVDt9S",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 89
        },
        "outputId": "fb7447c6-ae9d-43a5-a5da-e4a7c17f1752",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524324751171,
          "user_tz": 240,
          "elapsed": 24545,
          "user": {
            "displayName": "Daniel Briggs",
            "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s128",
            "userId": "112609433765816672561"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "#########Delete before turn in#####################\n",
        "#import dataset to google\n",
        "from google.colab import files\n",
        "uploaded = files.upload()\n",
        "#for name, data in uploaded.items():\n",
        "#  with open(name, 'wb') as f:\n",
        "#    f.write(data)\n",
        "#    print('saved file', name)  \n",
        "#####################################################"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-f2bbf2b5-759f-4167-ac85-4493b26e9dac\" name=\"files[]\" multiple disabled />\n",
              "     <output id=\"result-f2bbf2b5-759f-4167-ac85-4493b26e9dac\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving simulated_genos.txt to simulated_genos (1).txt\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "p-fFqqNDru5W",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          }
        }
      },
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import seaborn as sns"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "uyC8CvMFrw0Y",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "9912fe7f-de4c-42ba-828a-b9151f4ef0ce",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524324776843,
          "user_tz": 240,
          "elapsed": 1145,
          "user": {
            "displayName": "Daniel Briggs",
            "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s128",
            "userId": "112609433765816672561"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "G = np.loadtxt('simulated_genos.txt', dtype='i', delimiter=' ')\n",
        "G = np.asmatrix(G)\n",
        "print(np.linalg.matrix_rank(G))\n",
        "print(G.shape)"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "50\n",
            "(20000, 50)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "4yKeh597coTc",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Tentative Work"
      ]
    },
    {
      "metadata": {
        "id": "16fsdq9aDzYE",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          }
        }
      },
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import numpy.linalg as npl\n",
        "import scipy.linalg as scalg\n",
        "import pandas as pd\n",
        "from scipy import stats\n",
        "import math as math\n",
        "\n",
        "# Input: m x n genotype matrix with m individuals and n loci \n",
        "# First 10000 individuals are cases\n",
        "# Next 10000 individuals are controls\n",
        "#df = np.loadtxt('C:/Users/debri/Downloads/simulated_genos', dtype='i', delimiter=' ')\n",
        "#df = np.asmatrix(df)\n",
        "\n",
        "\n",
        "def fastSKAT(G): \n",
        "  # m individuals, n loci\n",
        "  m, n = G.shape \n",
        "  \n",
        "  # generate the design matrix\n",
        "  # null model assumption: intercept only model\n",
        "  # create m x 1 design matrix \n",
        "  # use design matrix to create matrix S \n",
        "  # definition of S: the projection orthogonal to the range of X\n",
        "  X = None \n",
        "  if X is None:\n",
        "      # design matrix\n",
        "      X = np.ones((m,1))\n",
        "      # next step to matrix S\n",
        "      XTX = npl.inv(np.matmul(np.transpose(X), X))\n",
        "      # next step to matrix S\n",
        "      XX = np.matmul(np.matmul(X,XTX),np.transpose(X))\n",
        "      # creates matrix S\n",
        "      S = np.diag([1]*m) - XX\n",
        "\n",
        "  # logit(y) = alphaX  \n",
        "  # logit(y) = alpha_0 \n",
        "  # p(y) = 0.5\n",
        "  Y = np.array([1]*10000 + [0]*10000)\n",
        "  Ybar = Y - 0.5\n",
        "\n",
        "  # Weight matrix generation\n",
        "  # Column means asymptotically are equivalent to the MAFj for each loci\n",
        "  a = np.array(G.mean(axis=0).tolist()[0])\n",
        "  # Fence post algorithm\n",
        "  # Initiate first row of the weight matrix\n",
        "  W = np.array(np.transpose(a))\n",
        "    \n",
        "  # Generate the weight matrix\n",
        "  # iteratively add a row for each individuals\n",
        "  for i in range(m - 1):\n",
        "    W = np.vstack((W, np.transpose(a)))\n",
        "\n",
        "  # Weighted genotype matrix G-bar \n",
        "  # G-bar is the Hadamard product of the genotype matrix and the weight matrix\n",
        "  Gbar = (np.transpose(np.multiply(W,G))  @ S)/math.sqrt(2)\n",
        "  \n",
        "  # matrix H: n x n matrix \n",
        "  # product of weighted genotype matrix and orthogonal projection to the range of X\n",
        "  H = np.matmul(Gbar,np.transpose(Gbar))\n",
        "  \n",
        "  #computation of eigenvalues \n",
        "  Eig = npl.eigvals(H)\n",
        "\n",
        "  # calculation of test statistic Q \n",
        "  # probability under null model (intercept only)\n",
        "  mu_hat= 0.5\n",
        "  # variance of Y - Y_bar\n",
        "  var_hat=np.var(Y-mu_hat)\n",
        "  # outer list for summation\n",
        "  outer = []\n",
        "  # loop for Q statistic evaluation\n",
        "  for j in range(n):\n",
        "    inner = []\n",
        "    for i in range(m):\n",
        "      inner.append(Ybar[i]*G[i,j])\n",
        "    outer.append(W[0,j] * 1/(2*var_hat) * np.power(sum(inner),2))\n",
        "  #  \n",
        "  Q = sum(outer)\n",
        "  return Q,Eig,H\n",
        "  \n",
        "\n",
        "Q,Eig,H=fastSKAT(G)  \n",
        "  \n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "wnIpnD10lsp2",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "82c742ce-cf5c-4959-9dc6-7145f5a8954a",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524331211937,
          "user_tz": 240,
          "elapsed": 536,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "def fastSKAT_Pvalue(Q,Eig,H):\n",
        "    #Basic Satterthwaite  \n",
        "    from scipy import stats\n",
        "\n",
        "    ###Calculate a and v for the chi-squared distribution\n",
        "    a= np.sum(np.square(Eig))/np.sum(Eig)\n",
        "    v = np.square(np.sum(Eig))/ np.sum(np.square(H))\n",
        "    print(\"Basic Satterthwaite approximations p-value:\",(stats.chi2.cdf(Q*a, v)))\n",
        "\n",
        "\n",
        "    #Improved Satterthwaite\n",
        "    ###largest eigenvalues index\n",
        "    k_index=Eig.argsort()[-10:][::-1]\n",
        "    eig_biggest=Eig[k_index]\n",
        "    H_small=H[-k_index]\n",
        "    eig =Eig[-k_index]\n",
        "    a_k= np.sum(np.square(eig))/np.sum(eig)\n",
        "    v_k = np.square(np.sum(eig))/ np.sum(np.square(H_small))\n",
        "    first_part=stats.chi2.cdf(np.sum(eig_biggest)*Q,1)\n",
        "    second_part=stats.chi2.cdf(Q*a_k , v_k)\n",
        "\n",
        "    print(\"Improved approximations p-value:\",(first_part+second_part))\n",
        "\n",
        "fastSKAT_Pvalue(Q,Eig,H)"
      ],
      "execution_count": 96,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Basic Satterthwaite approximations p-value: 1.4809326709532333e-56\n",
            "Improved approximations p-value: 0.032086095919380477\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "ab825nq6I5Ms",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 349
        },
        "outputId": "1e56604f-20af-41a9-8b96-68e9ac6b0dc9",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524330652512,
          "user_tz": 240,
          "elapsed": 423,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "U, S, V = np.linalg.svd(H) \n",
        "eigvals = S**2 / np.cumsum(S)[-1]\n",
        "\n",
        "num_vars = 50\n",
        "num_obs = 50\n",
        "fig = plt.figure(figsize=(8,5))\n",
        "sing_vals = np.arange(num_vars) + 1\n",
        "plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)\n",
        "plt.title('Scree Plot')\n",
        "plt.xlabel('Principal Component')\n",
        "plt.ylabel('Eigenvalue')\n",
        "leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, \n",
        "                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),\n",
        "                 markerscale=0.4)\n",
        "leg.get_frame().set_alpha(0.4)\n",
        "leg.draggable(state=True)\n",
        "plt.show()"
      ],
      "execution_count": 92,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgsAAAFMCAYAAABF6v+HAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBo\ndHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzs3X9cleX9x/HXfQ4cEDkoIMcg0/i6\nivwdc7kgNUPKWS0tf5JWy60sNC1aOtemW/4qk0qzsmbTLB1JVuYauhWtJies0dQ0V7YySaeHRBAB\ngcP5/qEeY57DDzuHA/h+Ph4+Hpz7uq/7fO5PPPLjdV33dRsul8uFiIiIiBemQAcgIiIiLZuKBRER\nEamXigURERGpl4oFERERqZeKBREREamXigURERGpl4oFEWmUTz75hNtuu41hw4Zx7bXXMnbsWD76\n6KNmj2P9+vX069ePYcOGMWzYMK655hruu+8+Dh8+DMDMmTN5+umnG7zOK6+84u9QRdoMFQsi0iCX\ny8XkyZP52c9+Rk5ODps2bWLSpEmkp6dTUVHR7PH069ePnJwc95+OHTvy8MMPN7q/w+HgD3/4gx8j\nFGlbVCyISIOKi4txOBz07dvXfeyaa67hjTfeoF27dgA899xzpKSkcO2117JgwQJcLhf5+fmMGzeO\nadOmkZGRAcDf/vY3brjhBlJSUrjjjjvcIwJVVVXMnTuXa6+9lquvvppnn322UbGZTCZuueUWtmzZ\nckbb7t27GTduHMOGDePGG2/k/fffB2DcuHHs37+fYcOGUVVV9b1yI3IuULEgIg2KjIykd+/e3Hrr\nraxbt459+/YBcN555wHw0UcfkZ2dzRtvvMGbb77JP//5T3JycgDYtWsX48aNY/Hixezbt48HH3yQ\nxYsX8/bbbzNgwADmzJkDwPPPP8+ePXt488032bhxI5s2bSI3N7dR8dXU1GCxWOocq62t5f7772fC\nhAnk5OQwd+5cMjIyKCsrY/78+cTGxpKTk3NGPxE5k4oFEWmQYRj88Y9/JDU1lRdffJGhQ4dy3XXX\nsXnzZgDee+89Bg8eTHh4OBaLhdWrV3PNNdcAEBoayhVXXOE+7/LLL+fiiy8GTvwL/5133sHpdJKb\nm0taWhoWi4WwsDBuvPFG9/XrU1VV5Y7tuwoLCykqKuK6664DoHfv3sTFxbFjxw6f5UXkXBEU6ABE\npHWwWq3ce++93HvvvRQVFbF+/Xruv/9+3njjDYqLi7HZbO5zT01NAHTo0MH989GjR/noo48YNmyY\n+1h4eDhHjhzh6NGjLFiwgMzMTOBEEdCnTx+PsfzrX/9yX8NkMnHFFVfwwAMP1Dnn8OHDWK1WDMNw\nH4uIiODw4cN06tTpe2RC5NyjYkFEGvTf//6XwsJC+vfvD0CnTp248847ycnJ4fPPPycyMpLi4mL3\n+d/9+btsNhtJSUksWbLEY9sdd9zBkCFDGoynX79+rFy5st5zoqOjKSkpweVyuQuGI0eOEB0d3eD1\nRaQuTUOISIMOHDhAeno6n3zyifvY9u3b2b9/P7179+bqq6/mnXfeoaSkhJqaGtLT0/nHP/5xxnWu\nvPJKPvroI/eah+3btzN37lwAUlJSWLduHU6nE5fLxdNPP81777131jF36dKF8847j7feeguAgoIC\nioqK6NOnD0FBQZSXl1NTU3PW1xc5l2hkQUQadNlll/Hwww8zZ84cjh49Sm1tLZ06deLxxx/n/PPP\n5/zzz2fSpEmMGDECi8XCwIEDuf7669m6dWud69hsNh5++GHS09Oprq6mffv2zJo1C4C0tDQKCwu5\n7rrrcLlc9OrVi9tuu+2sYzYMg8zMTGbPns1TTz1Fu3btePLJJwkLC+OSSy6hQ4cOJCcn89prrxEX\nF/e98iPS1hkul8sV6CBERESk5dI0hIiIiNTLr9MQ8+fPZ9u2bRiGwaxZs+qsbM7LyyMzMxOz2cyg\nQYNIT0/32ufAgQM8+OCDOJ1OYmJiWLRoERaLhQ0bNrBq1SpMJhNjxoxh9OjRVFdXM3PmTPbv34/Z\nbGbBggVccMEFbNq0iRdeeIHg4GA6d+7MggUL9Hy1iIhIY7j8JD8/33XnnXe6XC6Xa8+ePa4xY8bU\naf/JT37i2r9/v8vpdLrGjx/v+vzzz732mTlzpuutt95yuVwu1+LFi10vv/yy69ixY65rrrnGVVpa\n6qqoqHBdd911ruLiYtf69etdc+bMcblcLtf777/vmjZtmsvlcrmuvPJKV2lpqcvlcrkeeugh18aN\nG/116yIiIm2K36Yh7HY7Q4cOBaB79+6UlJRQVlYGwL59++jQoQOxsbGYTCYGDx6M3W732ic/P5+U\nlBQAhgwZgt1uZ9u2bfTu3Rur1UpoaCiJiYkUFBRgt9vdm7MkJSVRUFAAQMeOHSktLQWgtLSUyMhI\nf926iIhIm+K3YqGoqKjOX8hRUVE4HA7gxEtcoqKizmjz1qeiosI9ZRAdHe0+19s1Th03mUwYhkFV\nVRUPPfQQI0eOJCUlhdraWpKSkvx16yIiIm1Ksy1wdJ3FQxee+ni7TkPH586dS3Z2Nn/7298wmUy8\n/fbb9X53TY2zidGKiIi0TX5b4Giz2SgqKnJ/PnToEDExMR7bDh48iM1mIzg42GOfsLAwKisrCQ0N\ndZ/r6fr9+vXDZrPhcDhISEiguroal8vlnn7o2rUrAFdccQWffPKJe2rDk+Li8ibdb0yMFYfjaJP6\niGfKpe8ol76jXPqOcukb/shjTIzV43G/jSwkJyezadMmAHbu3InNZiM8PBw4sbNaWVkZhYWF1NTU\nkJubS3Jystc+SUlJ7uObN29m4MCB9O3blx07dlBaWsqxY8coKCigf//+JCcnu992l5uby4ABA4iM\njKSkpMT9KtwdO3bQrVs3f926iIhIm+K3kYXExER69uzJuHHjMAyD2bNns379eqxWK6mpqcyZM8f9\nfvvhw4cTHx9PfHz8GX0Apk6dyowZM8jKyiIuLo4RI0YQHBxMRkYGkyZNwjAM0tPTsVqtDB8+nLy8\nPMaPH4/FYmHhwoWYzWZ++9vfMnnyZCwWC126dHG/iU5ERETqpx0cvWjq0I6G1XxHufQd5dJ3lEvf\nUS59o01MQ4iIiEjboGJBRERE6qViQUREROqlV1SLiIhfHDiwn1tvHccllyTUOb58+TOsXr2ayy5L\npFevPl56+15BwUesX/8Kc+c+6tPrZmY+wiefbGfp0uW0bx/u02s///wzfPhhPhaLBaezhvvvn8Hh\nw4d59dVX+OMf/+A+79ixMm65ZTTZ2W+SkpJM7959gRN7Dd1002hSUq75XnGoWGgGIa9lE/bEYsyf\n7cZ5cQLl0zM4PnJUoMMSEfG7rl278dRTz9U51rGjlYkTbw9MQH5gt+fxwgsv+bxQ+Pjjf/L55/9m\n+fI/YhgGBQUf8fLLL/Kb3/yeRx6Ze3IPIQOA99//O0lJVxIUFER4eLg754cPf8vMmRm0bx/Oj398\n9jsXq1jws5DXsom46w7356BPdxJx1x2UggoGETlnzZs3h6uuSqFv38t46KEHOX78OFdckcybb77O\nunUb2LbtY5YvX0ZQUBA2W2dmzHiIHTu2sX79KxiGib17v+Sqq1IYOPAqli7NZMmSZwF44YXnsFoj\nuPDCeP7wh2cJDg7GarXy+98vrPP9112Xwp//fGIn34ceepCbbhpDQsKlzJ//O44ePYrT6WT69F/y\ngx9cxEsvreTvf8/FZDKRnDyQW289/f/0NWte5NtvHcyYcR/jx0/gzTdfp7y8nClT7uObbwrJynoZ\ns9nMJZdcyvTpD7BixXJKSo5QWFjI/v3f8Itf3M2f/7yB//53P4sWPcn553dxX7us7CgVFRU4nU6C\ngoJITOxPYmJ/AAYOHMzf/vY3Bg488S6kd975G+PHTzgjz1FR0UyZMp2VK//wvYoFrVnws7AnFns+\n/mRmM0ciIueyiLRRxNgifPonIu37/4MnJ2cjF174fzzzzArCw63uLfqfeGIRCxcuZsmSZ4mKiiI3\n928A7Nq1k1//eg7PPvtHXn01i4suupiiIgdHj554hPAf/3iPq666mqNHjzJ79lyeeuo5wsLak59v\nbzCWV15Zy4ABSTz55DNkZMzkqaceB+BPf3qJZ55ZwbPPvoDVGlGnT1rarURFRfPYY0to3z6cL77Y\nQ2bmU3Tt2o3nnlvGE088zTPPrGD//m8oKPgIOPEyw8zMpQwZkkJOzsaTPw9ly5b36lx7wIAkzGYz\nY8bcyKJF87Hbt7jzk5o6jL/85S8AlJWVsXfvl/Trl+jxvhISevDVV1826r+HNxpZ8DPzZ7ubdFxE\npC35+uu9TJlyp/tz167dWLTo9L/yv/rqKy677IcAXHnlINaseZHDh7+lsHAfs2b9EoDKyko6dOhI\np04xXHJJAqGhoXW+Izl5EPn5efTq1ZeQEAsxMTY6duzII4/Mxel0sn//N/zwhz8iLCys3lh37NjO\nkSPFbNr0FgDHj1cCcNVVKUyffg+pqcO45pph9V7jBz+4CIvFwpdf/ocuXbq6v/Oyy37IZyf/v9+j\nR08AOnXqhGGcmEaIioqipKSkzrUsFgtPPPE0u3fv4sMP81m6NJO3397MQw/9jl69+vD1119TWlrC\nli3vM2jQEPe1/ld5eTkm0/cbG1Cx4GfOixMI+nSnx+MiIs2ldE12o84LXbGcsGeXUT55CpWT7my4\nQwM8rVmoy4XJdOIvuVN/2QUFBdOpU8wZ/QoKPsJsNp9xhcGDh/Dqq69QUnKEwYOvBmDBgodZtOgJ\nLrwwnszMR+qNsaamBoDg4CDuu++XZyy6fOCBX7F371e8885fmTr1Lp57bhVBQZ7/+gwODj55L3Vf\ncFhTU01ISAhAnXv47s//u0ei0+nE5XKRkNCDhIQejBo1jpEjf4LT6cRsNjNs2DDee+9d3nvvXX7+\n87u83t/u3bu46KJL6s1BQzQN4Wfl0zM8H592fzNHIiLSsMpJd3H4w+0+KRQaIy6uC7t3fwrABx/k\nARARcWKo/8sv/wNAdvaf2LPnc6/X6NmzN1999R/y8rZw1VVDgRNPB3TufB5Hjx6loOCfVFdX1+lj\nGAaVlZVUVlby2Wf/BqBHj16899677u/+059eoqysjD/+8Xm6dbuQn/3sF1itHSgvP9bgfV1wQTcK\nC792n/vxxwVcckmPxqYFgBUrlvPCC6cLpiNHiomKinYXGNdffz25uW9TVHSIi738A7S4+DDLlz/1\nvReUamTBz46PHEWp04n1nl9gADWX9qB8+gNa3Cgi54T/nYYA+PWvf+X+efjwG/jVr+5nypQ7+dGP\nBriHy2fO/C3z5/+O4OAToww//elNfPLJdo/fYRgGvXr15fPP/815550HwE03jebuuydxwQVdueWW\nW3nhhee488573H1GjBjFnXfexoUX/h+XXHIpAKNGjWXevDncc8/Pqa2tZfr0BwgPD+fIkWJ+8Ytb\nadcujF69+hAR0aHB+27Xrh3p6dPIyJiKYZjo06cfffv246OP8hudu1tvvYPMzEe4887badeuHbW1\ntfz6179zt1900UV8+20RV111dZ1+ZWVlTJlyJ05nDZWVlYwfP5EePXo1+ns90bshvPD1uyGi+vfB\n/PVXHLb/E2f3i75veG2a9o33HeXSd5RL3/luLv/73wPs3fsVAwZcwSefbGfFiuU8/viyAEfYOjTn\nuyE0stBMnF27Yv76K0x796pYEBE5qX37cLKyXmblyudxuWD69AcCHZJ4oGKhmTi7dgPAvO9rqhs4\nV0TkXGG1WsnMfCrQYUgDtMCxmdRe0BUA89d7AxyJiIhI06hYaCanRhZM+1QsiIhI66JioZk4Lzg5\nDaGRBRERaWVULDST2m6nioWvAxyJiIhI06hYaCa1nc/DZbFgKnLAsYY39BAREWkpVCw0F5MJZ5cL\nADAX7gtwMCIiIo2nYqEZnX4i4qvABiIiItIEKhaakbPrhQCYtG5BRERaERULzcjZVXstiIhI66Ni\noRnVfmcXRxERkdbCr9s9z58/n23btmEYBrNmzaJPn9PvCM/LyyMzMxOz2cygQYNIT0/32ufAgQM8\n+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\nCCGs+n+mACLOCBdmYAAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<matplotlib.figure.Figure at 0x7fdfe03349b0>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "metadata": {
        "id": "7XeJaMTwMzw0",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          }
        }
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "pfRaXtIxrofL",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 292
        },
        "outputId": "fedd21b8-9c7b-4db3-9eb2-a5d2ba4fcdf5",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524324726519,
          "user_tz": 240,
          "elapsed": 273,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "# splitting data into cases/controls\n",
        "cases = G[0:10000]\n",
        "controls = G[10000:]\n",
        "print('cases\\n',cases[0:3])\n",
        "print('controls\\n',controls[0:3])\n",
        "y = np.array([1]*10000 + [0]*10000).reshape((20000,1))\n",
        "y.shape\n"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "cases\n",
            " [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]]\n",
            "controls\n",
            " [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(20000, 1)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 16
        }
      ]
    },
    {
      "metadata": {
        "id": "eEOpVGobGmzt",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "543ea1b8-17b0-4083-8800-1aedf57c8739",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524322398852,
          "user_tz": 240,
          "elapsed": 352,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "#Weight matrix W: apparently standard form is $W_{i,i}$\n",
        "\n",
        "a = np.array([1/np.shape(G)[1]]*np.shape(G)[1])\n",
        "W = np.diag(a)\n",
        "print(W)"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[0.02 0.   0.   ... 0.   0.   0.  ]\n",
            " [0.   0.02 0.   ... 0.   0.   0.  ]\n",
            " [0.   0.   0.02 ... 0.   0.   0.  ]\n",
            " ...\n",
            " [0.   0.   0.   ... 0.02 0.   0.  ]\n",
            " [0.   0.   0.   ... 0.   0.02 0.  ]\n",
            " [0.   0.   0.   ... 0.   0.   0.02]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "yMeApp76CcfX",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 71
        },
        "outputId": "cfc1a5c0-40a7-44ea-b16a-14e961166983",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524328582536,
          "user_tz": 240,
          "elapsed": 1579,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "import statsmodels.api as sm\n",
        "#m=sm.Logit(y,G)\n",
        "#result=m.fit()\n",
        "#print(result.summary())\n",
        "#print(result.model)"
      ],
      "execution_count": 71,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.\n",
            "  from pandas.core import datetools\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "metadata": {
        "id": "jJR8yG8tSi8-",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "outputId": "973f432f-9bfd-4606-a666-a0136c4f6b2e",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524322406556,
          "user_tz": 240,
          "elapsed": 5055,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "\n",
        "normed = (G - G.mean(axis=0)) / G.std(axis=0)\n",
        "print(normed.mean(axis = 0))\n",
        "print(normed.std(axis = 0))\n",
        "\n",
        "ZW = np.matmul(normed,W)\n",
        "K = np.matmul(ZW, np.transpose(normed))\n",
        "\n"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[ 6.25973925e-16  1.75802203e-15 -9.03263575e-16 -9.10556353e-17\n",
            "  -2.81889095e-16  1.28511438e-16 -4.24349098e-16 -6.90680846e-16\n",
            "  -4.15809748e-16  2.05119949e-16  1.14757127e-15  3.31422043e-16\n",
            "   2.98003844e-15  2.61273295e-15  1.94080862e-16  6.15651280e-16\n",
            "  -1.17515997e-15  9.07529260e-16  4.57841577e-16 -3.47704504e-17\n",
            "  -2.87452701e-16  6.92301078e-16  1.20261578e-15  6.56488752e-17\n",
            "   5.81086915e-16  1.75787163e-15 -8.52651283e-16  2.01031553e-16\n",
            "  -9.39240352e-16  2.93200186e-16  7.35620939e-16 -2.66304340e-17\n",
            "   7.13019227e-16 -4.21351148e-16  1.40239209e-16  1.56708674e-16\n",
            "  -5.38203856e-16  1.50455516e-15 -4.30752656e-17 -1.32587899e-15\n",
            "   5.43034714e-15  7.95719567e-16  3.11502907e-16 -1.21782445e-16\n",
            "  -7.14650561e-17  1.38179225e-15  1.45805867e-15  3.28705552e-16\n",
            "  -9.63273558e-16  2.83455898e-16]]\n",
            "[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n",
            "  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n",
            "  1. 1.]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "z0g4Q6nFTgrD",
        "colab_type": "code",
        "colab": {
          "autoexec": {
            "startup": false,
            "wait_interval": 0
          },
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "728108e4-adf0-4faf-98d0-ff04b7672939",
        "executionInfo": {
          "status": "ok",
          "timestamp": 1524322431098,
          "user_tz": 240,
          "elapsed": 651,
          "user": {
            "displayName": "Yueting Luo",
            "photoUrl": "//lh5.googleusercontent.com/-_7IwQZGsigs/AAAAAAAAAAI/AAAAAAAAADM/6cjNZBuIHB0/s50-c-k-no/photo.jpg",
            "userId": "107659994744764221986"
          }
        }
      },
      "cell_type": "code",
      "source": [
        "print(np.ndim(K))\n",
        "yTk = np.matmul(np.transpose(y),K)\n",
        "yTky = np.matmul(yTk,y)\n",
        "print(yTky.shape)\n",
        "\n",
        "\n",
        "#np.linalg.svd(K[:500])\n"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2\n",
            "(1, 1)\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}