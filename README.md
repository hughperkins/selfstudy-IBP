# selfstudy-IBP
Self-study notes for Indian Buffet Process, from reading through "The Indian Buffet Process: An Introduction and Review", Griffiths, Ghahramani, 2011

How this is organized currently:

The following sections go through Griffiths and Ghahramani tutorial, in sequence.  Occasionally, I found I needed
some additional resources/information, to understand the Griffiths and Ghahramani tutorial, and interlude to such
other resources, in the middle of some of these sections.

- [ibp_section1.ipynb](ibp_section1.ipynb) revision of multinomial distributions, Dirichlet distributions, mixture models
- [ibp_section2.ipynb](ibp_section2.ipynb) Latent class models
- [ibp_section3.ipynb](ibp_section3.ipynb) Latent feature models
- [ibp_section4.ipynb](ibp_section4.ipynb) Example: Linear-Gaussian Latent Feature Model with Binary Features (in progress)

Some notebooks are apart from the above sections, since they use a ton of browser memory, so I've separated them out:

- [sampling from prior.ipynb](sampling from prior.ipynb)  notebook trying some simple sampling from an IBP prior, without any data
- [Demonstration on 6x6 images.ipynb](Demonstration on 6x6 images.ipynb) notebook trying to sample from the posterior, with some simple toy data (from the Griffiths and Ghahramani tutorial).  This is in progress for now

I also separated out my reading through Doshi-Velez's "accelerated sampling" paper, because it was also using a lot of browser memory.  Actually, this will probably be moved to a separate repo at some point, probably.

- [accelerated_gibbs_sampling.ipynb](accelerated_gibbs_sampling.ipynb) (This is in-progress, for now)
