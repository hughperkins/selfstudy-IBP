# selfstudy-IBP

These notes adapted from:

- ["The Indian Buffet Process: An Introduction and Review"](http://jmlr.org/papers/volume12/griffiths11a/griffiths11a.pdf), Griffiths, Ghahramani, 2011
- ["Introduction to the Dirichlet Distribution and Related Processes"](http://mayagupta.org/publications/FrigyikKapilaGuptaIntroToDirichlet.pdf), Frigyik, Kapila, Gupta
- ["Advanced Data Analysis from an Elementary Point of View"](http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/), Cosma Rohilla Shalizi, chapter 19, "Mixture Models"
- ["Mixture Models and the EM Algorithm"](http://mlg.eng.cam.ac.uk/tutorials/06/cb.pdf), slide presentation by Bishop 2006
- ["Accelerated Sampling for the Indian Buffet Process"](http://mlg.eng.cam.ac.uk/pub/pdf/DosGha09.pdf), Doshi-Velez and Ghahramani

The tutorial by Griffiths and Gahramani above was my primary resouce.  Then, in order to understand it, I needed to reach out to the other resources above :-)

Generally speaking, these notes assume that you are reading the appropriate tutorial/paper/slides in parallel with these notes.
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
