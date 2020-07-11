from devito.tools import memoized_meth
from devito import VectorTimeFunction, TensorTimeFunction

from examples.seismic import Receiver
from elastic_VTI_solvers import ForwardOperator


class ElasticVTIWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        
        
        self.model = model
        self.geometry = geometry

        assert self.model == geometry.model

        self.space_order = space_order
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, v=None,tau=None, vp=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        tau : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        vp : Function or float, optional
            The time-constant velocity.
        save : int or Buffer, optional
            Option to store the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create Create Forward wavefield

        v = v or VectorTimeFunction(name='v', grid=self.model.grid, save=self.geometry.nt,
                               space_order=self.space_order, time_order=1)
        tau = tau or TensorTimeFunction(name='tau', grid=self.model.grid, save=self.geometry.nt,
                                 space_order=self.space_order, time_order=1)
        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, v=v, tau=tau, vp=vp,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, v, tau, summary
