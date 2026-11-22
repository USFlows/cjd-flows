API reference
=============

The v1 API. Direction convention for bijections: ``Bijection.forward`` maps
data to latent (the normalizing direction used for likelihoods),
``Bijection.inverse`` maps latent to data (the generative direction). On a
:class:`~cjd_flows.core.Flow` the corresponding methods are
:meth:`~cjd_flows.core.Flow.transform` (data → latent) and
:meth:`~cjd_flows.core.Flow.generate` (latent → data); calling the flow itself,
``flow(x)``, returns ``log_prob(x)``.

.. toctree::
   :maxdepth: 1

   package
   core
   models
   bijections
   conditioners
   radial
   train
   data
   scores
   evaluation
   export
   testing

..
   io  (save/load, added separately)
