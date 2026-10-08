# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:28:36 2024

@author: ybarhate
"""


import unittest
import os
import numpy as np
from copy import deepcopy
from PharmaPy.Reactors import BatchReactor, SemibatchReactor, PlugFlowReactor
from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import BatchCryst, MSMPR
from PharmaPy.SolidLiquidSep import Filter

from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.MixedPhases import SlurryStream

from PharmaPy.Kinetics import RxnKinetics, CrystKinetics

from PharmaPy.Utilities import CoolingWater
from PharmaPy.Interpolation import PiecewiseLagrange
from PharmaPy.ProcessControl import DynamicInput

from PharmaPy.SimExec import SimulationExec

from PharmaPy.Streams_Refactored import LiquidStream as NewLiquidStream
from PharmaPy.Phases_Refactored import LiquidPhase as NewLiquidPhase
from PharmaPy.Reactors_Refactored import (SemiBatchReactor as NewSemiReactor,
                                          ContinuousReactor as NewContReactor)
from PharmaPy.ProcessControl_Refactored import (SimpleTemperatureController,
                                                ContinuousVesselController)
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Commons import trapezoidal_rule


class TestFlowsheets(unittest.TestCase):
    """ Class containing code for testing flowsheet executions in PharmaPy
    """
          
    def test_B_BC_FILT(self):
        def make_non_verbose(runargs):
            for key in runargs:
                runargs[key]['verbose'] = False       
            return runargs
        # Reactor definition
        current_path = os.path.dirname(os.path.abspath(__file__))
        path_phys = os.path.join(current_path, 'data', 'compound_database.json')
        graph = 'R01 --> CR01 --> F01'
        flst = SimulationExec(path_phys, flowsheet=graph)
        rxns = ['A + B --> C','C + A --> D']
        k_vals = np.array([2.654e4, 5.3e2])  # Pre-exponential factor
        ea_vals = np.array([4.0e4, 3.0e4])  # Activation Energies
        kinetics = RxnKinetics(path=path_phys, rxn_list=rxns, k_params=k_vals, ea_params=ea_vals)
        temp_init = 313. # Temperature in Kelvin
        mole_conc_init = np.array([0.33, 0.33, 0, 0, 0]) # Equimolar concentration of A, B (mol/L)
        liquid_init = LiquidPhase(path_phys, temp = temp_init, mole_conc=mole_conc_init, vol=0.06, name_solv='solvent')
        temp_R01 = 313.15
        cw = CoolingWater(mass_flow=0.01, temp_in=temp_R01)  # mass flow in kg/s
        time_R01 = 3600.0 # Batch time in seconds
        flst.R01 = BatchReactor(isothermal=False)
        flst.R01.Utility = cw
        flst.R01.Phases = liquid_init
        flst.R01.Kinetics = kinetics
        
        # Crystallizer definition
        prim = (3e8, 0, 3)  # kP in #/m3/s
        sec = (4.46e10, 0, 2, 1e-5) # kS in #/m3/s
        growth = (5, 0, 1.32)  # kG in um/s
        dissol = (1, 0, 1) # kD in um/s
        solub_cts = np.array([2.269e2, -1.88e0, 3.89e-3])
        x_gr = np.geomspace(1, 1500, num=35)  # discretization point for crystal size distribution
        distrib_init = np.zeros_like(x_gr)
        massfrac_solid= [0, 0, 1, 0, 0]  # solid phase composed only by API (C)
        solid_cry = SolidPhase(path_phys, x_distrib=x_gr, distrib=distrib_init, mass_frac=massfrac_solid)
        temp_program = np.array([[313.15,308],
                         [308,295],
                         [295,278.15],
                        ], dtype=np.float64)
        time_CR01 = time_R01 * 2.0
        lagrange_fn = PiecewiseLagrange(time_CR01, temp_program)
        flst.CR01 = BatchCryst(target_comp='C', method='1D-FVM', scale=1e-9,
                       controls={'temp': lagrange_fn.evaluate_poly})
        flst.CR01.Kinetics = CrystKinetics(solub_cts, nucl_prim= prim, nucl_sec= sec,
                                   growth= growth, dissolution=dissol)
        flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)
        flst.CR01.Phases = solid_cry
        
        # Filter definition
        alpha = 1e11
        Rm = 1e10
        deltaP = 101325 # In Pa
        filt_area = 200  # cm**2
        diam = np.sqrt(4/np.pi * filt_area) / 100  # m
        flst.F01 = Filter(diam, alpha, Rm)
        
        # Solving the Flowsheet
        runargs_R01 = {'runtime': time_R01} # Runtime of the reactor
        sundials = {'maxh': 60} # Maximum step size we take in time is 1 minute [Based on experience for crystallization]0
        runargs_CR01 = {'runtime': time_CR01, 'sundials_opts': sundials} # Runtime
        runargs_F01 = {'runtime': None, 'deltaP':deltaP}  # Note that deltaP is passed here
        run_kwargs = {'R01': runargs_R01,
              'CR01': runargs_CR01,
              'F01': runargs_F01
              }
        run_kwargs = make_non_verbose(run_kwargs)
        flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)
        print('B-BC-FILT flowsheet simulation complete')
        
        
    def test_SB_BC_FILT(self):
        def make_non_verbose(runargs):
            for key in runargs:
                runargs[key]['verbose'] = False       
            return runargs
        # Reactor definition
        current_path = os.path.dirname(os.path.abspath(__file__))
        path_phys = os.path.join(current_path, 'data', 'compound_database.json')
        graph = 'R01 --> CR01 --> F01'
        flst = SimulationExec(path_phys, flowsheet=graph)
        rxns = ['A + B --> C','C + A --> D'] 
        k_vals = np.array([2.654e4, 5.3e2])  # Pre-exponential factor
        ea_vals = np.array([4.0e4, 3.0e4])  # Activation Energies
        kinetics = RxnKinetics(path=path_phys, rxn_list=rxns, k_params=k_vals, ea_params=ea_vals)
        temp_init = 313. # Temperature in Kelvin
        mole_conc_init = np.array([0.0, 0.66, 0.0, 0.0, 0.0]) # Only solvent and species B initially.
        vol_liq = 0.03
        liquid_init = LiquidPhase(path_phys, temp = temp_init, mole_conc=mole_conc_init, vol=vol_liq, name_solv='solvent') 
        temp_inlet = 313.15
        conc_inlet = np.array([0.66, 0.0, 0.0, 0.0, 0.0]) # Inputs in mol/L. (Solvent conc calculated internally)
        time_R01 = 7200.0 # Batch time in seconds
        flowrate_in = vol_liq/time_R01 # Flowrate in (m3/s)
        flow_inlet = LiquidStream(path_phys, temp=temp_inlet, mole_conc=conc_inlet, vol_flow=flowrate_in, name_solv='solvent')
        temp_R01 = 313.15
        cw = CoolingWater(mass_flow=0.01, temp_in=temp_R01)
        tank_vol = 0.1
        flst.R01 = SemibatchReactor(isothermal=False, vol_tank=tank_vol)
        flst.R01.Utility = cw
        flst.R01.Phases = liquid_init
        flst.R01.Kinetics = kinetics
        flst.R01.Inlet = flow_inlet
        # Crystallizer definition
        prim = (3e8, 0, 3)  # kP in #/m3/s
        sec = (4.46e10, 0, 2, 1e-5) # kS in #/m3/s
        growth = (5, 0, 1.32)  # kG in um/s
        dissol = (1, 0, 1) # kD in um/s
        solub_cts = np.array([2.269e2, -1.88e0, 3.89e-3])
        x_gr = np.geomspace(1, 1500, num=35)  # discretization point for crystal size distribution
        distrib_init = np.zeros_like(x_gr)
        massfrac_solid= [0, 0, 1, 0, 0]  # solid phase composed only by API (C)
        solid_cry = SolidPhase(path_phys, x_distrib=x_gr, distrib=distrib_init, mass_frac=massfrac_solid)
        temp_program = np.array([[313.15,308],
                         [308,295],
                         [295,278.15],
                        ], dtype=np.float64)
        time_CR01 = time_R01 * 2.0
        lagrange_fn = PiecewiseLagrange(time_CR01, temp_program)
        flst.CR01 = BatchCryst(target_comp='C', method='1D-FVM', scale=1e-9,
                       controls={'temp': lagrange_fn.evaluate_poly})
        flst.CR01.Kinetics = CrystKinetics(solub_cts, nucl_prim= prim, nucl_sec= sec,
                                   growth= growth, dissolution=dissol)
        flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)
        flst.CR01.Phases = solid_cry
        
        # Filter definition
        alpha = 1e11
        Rm = 1e10
        deltaP = 101325 # In Pa
        filt_area = 200  # cm**2
        diam = np.sqrt(4/np.pi * filt_area) / 100  # m
        flst.F01 = Filter(diam, alpha, Rm)
        
        # Solving the Flowsheet
        runargs_R01 = {'runtime': time_R01} # Runtime of the reactor
        sundials = {'maxh': 60} # Maximum step size we take in time is 1 minute [Based on experience for crystallization]0
        runargs_CR01 = {'runtime': time_CR01, 'sundials_opts': sundials} # Runtime
        runargs_F01 = {'runtime': None, 'deltaP':deltaP}  # Note that deltaP is passed here
        run_kwargs = {'R01': runargs_R01,
              'CR01': runargs_CR01,
              'F01': runargs_F01
              }
        run_kwargs = make_non_verbose(run_kwargs)
        flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)
        print('SB-BC-FILT flowsheet simulation complete')
        
    def test_PFR_HOLD_BC_FILT(self):
        def make_non_verbose(runargs):
            for key in runargs:
                runargs[key]['verbose'] = False       
            return runargs
        # Reactor definition
        current_path = os.path.dirname(os.path.abspath(__file__))
        path_phys = os.path.join(current_path, 'data', 'compound_database.json')
        graph = 'R01 --> HOLD01 --> CR01 --> F01'
        flst = SimulationExec(path_phys, flowsheet=graph)
        rxns = ['A + B --> C','C + A --> D'] 
        k_vals = np.array([2.654e4, 5.3e2])  # Pre-exponential factor
        ea_vals = np.array([4.0e4, 3.0e4])  # Activation Energies
        kinetics = RxnKinetics(path=path_phys, rxn_list=rxns, k_params=k_vals, ea_params=ea_vals)
        vol_liq = 0.010  # m**3
        tau_R01 = 1800  # s
        vol_flow = vol_liq / tau_R01  # m**3 / s
        w_init = np.array([0, 0, 0, 0, 1])  # mass fraction # Only solvent is present at the start 
        liquid_init = LiquidPhase(path_phys, temp = 313.15, mass_frac=w_init, vol=vol_liq) # This is for the initial holdup existing in PFR
        temp_R01 = 313.15
        cw = CoolingWater(mass_flow=0.1, temp_in=temp_R01) 
        c_in = np.array([0.33, 0.33, 0, 0, 0])
        temp_in = 40 + 273.15  # K
        liquid_in = LiquidStream(path_phys, temp_in, mole_conc=c_in, vol_flow = vol_flow, name_solv='solvent')
        diam_in = 1 / 2 * 0.0254  # 1/2 inch in m
        flst.R01 = PlugFlowReactor(diam_in=diam_in, num_discr=50, isothermal=False)
        flst.R01.Utility = cw
        flst.R01.Phases = liquid_init
        flst.R01.Inlet = liquid_in
        flst.R01.Kinetics = kinetics
        time_R01 = 3600 *2
        
        flst.HOLD01 = DynamicCollector()
        
        # Crystallizer definition
        prim = (3e8, 0, 3)  # kP in #/m3/s
        sec = (4.46e10, 0, 2, 1e-5) # kS in #/m3/s
        growth = (5, 0, 1.32)  # kG in um/s
        dissol = (1, 0, 1) # kD in um/s
        solub_cts = np.array([2.269e2, -1.88e0, 3.89e-3])
        x_gr = np.geomspace(1, 1500, num=35)  # discretization point for crystal size distribution
        distrib_init = np.zeros_like(x_gr)
        massfrac_solid= [0, 0, 1, 0, 0]  # solid phase composed only by API (C)
        solid_cry = SolidPhase(path_phys, x_distrib=x_gr, distrib=distrib_init, mass_frac=massfrac_solid)
        temp_program = np.array([[313.15,308],
                         [308,295],
                         [295,278.15],
                        ], dtype=np.float64)
        time_CR01 = time_R01 * 2.0
        lagrange_fn = PiecewiseLagrange(time_CR01, temp_program)
        flst.CR01 = BatchCryst(target_comp='C', method='1D-FVM', scale=1e-9,
                       controls={'temp': lagrange_fn.evaluate_poly})
        flst.CR01.Kinetics = CrystKinetics(solub_cts, nucl_prim= prim, nucl_sec= sec,
                                   growth= growth, dissolution=dissol)
        flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)
        flst.CR01.Phases = solid_cry
        
        # Filter definition
        alpha = 1e11
        Rm = 1e10
        deltaP = 101325 # In Pa
        filt_area = 200  # cm**2
        diam = np.sqrt(4/np.pi * filt_area) / 100  # m
        flst.F01 = Filter(diam, alpha, Rm)
        
        # Solving the Flowsheet
        runargs_R01 = {'runtime': time_R01} # Runtime of the reactor
        sundials = {'maxh': 60} # Maximum step size we take in time is 1 minute [Based on experience for crystallization]0
        runargs_hold = {'runtime': time_R01}
        runargs_CR01 = {'runtime': time_CR01, 'sundials_opts': sundials} # Runtime
        runargs_F01 = {'runtime': None, 'deltaP':deltaP}  # Note that deltaP is passed here
        run_kwargs = {'R01': runargs_R01,
          'HOLD01': runargs_hold,
          'CR01': runargs_CR01,
          'F01': runargs_F01}
        run_kwargs = make_non_verbose(run_kwargs)
        flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)
        print('PFR_HOLD_BC_FILT flowsheet simulation complete')
         
    def test_PFR_MSMPR_HOLD_FILT(self):
        def make_non_verbose(runargs):
            for key in runargs:
                runargs[key]['verbose'] = False       
            return runargs
        # Reactor definition
        current_path = os.path.dirname(os.path.abspath(__file__))
        path_phys = os.path.join(current_path, 'data', 'compound_database.json')
        graph = 'R01 --> CR01 --> HOLD01 --> F01'
        flst = SimulationExec(path_phys, flowsheet=graph)
        rxns = ['A + B --> C','C + A --> D'] 
        k_vals = np.array([2.654e4, 5.3e2])  # Pre-exponential factor
        ea_vals = np.array([4.0e4, 3.0e4])  # Activation Energies
        kinetics = RxnKinetics(path=path_phys, rxn_list=rxns, k_params=k_vals, ea_params=ea_vals)
        vol_liq = 0.010  # m**3
        tau_R01 = 1800  # s
        vol_flow = vol_liq / tau_R01  # m**3 / s
        w_init = np.array([0, 0, 0, 0, 1])  # mass fraction # Only solvent is present at the start 
        liquid_init = LiquidPhase(path_phys, temp = 298, mass_frac=w_init, vol=vol_liq) # This is for the initial holdup existing in PFR
        cw = CoolingWater(mass_flow=0.1, temp_in=313.15)
        c_in = np.array([0.33, 0.33, 0, 0, 0])
        temp_in = 273.15 + 40
        liquid_in = LiquidStream(path_phys, temp_in, mole_conc=c_in, vol_flow = vol_flow, name_solv='solvent')
        diam_in = 1 / 2 * 0.0254  # 1/2 inch in m
        flst.R01 = PlugFlowReactor(diam_in=diam_in, num_discr=50, isothermal=False)
        flst.R01.Utility = cw
        flst.R01.Phases = liquid_init
        flst.R01.Inlet = liquid_in
        flst.R01.Kinetics = kinetics
        time_R01 = 3600 *10
        # Crystallizer definition
        prim = (3e8, 0, 3)  # kP in #/m3/s
        sec = (4.46e10, 0, 2, 1e-5) # kS in #/m3/s
        growth = (5, 0, 1.32)  # kG in um/s
        dissol = (1, 0, 1) # kD in um/s
        solub_cts = np.array([2.269e2, -1.88e0, 3.89e-3])
        cryst_kin =  CrystKinetics(solub_cts, nucl_prim= prim, nucl_sec= sec,
                                   growth= growth, dissolution=dissol)
        liq_vol = 0.01
        temp_init_cry = 313.15
        c_init = np.array([0,0,cryst_kin.get_solubility(temp_init_cry),0,0])
        liquid_init_cry = LiquidPhase(path_phys, vol=vol_liq, mass_conc=c_init, temp=temp_init_cry, name_solv = 'solvent')
        x_gr = np.geomspace(1, 1500, num=35)  # discretization point for crystal size distribution
        distrib_init = np.zeros_like(x_gr)
        massfrac_solid= [0, 0, 1, 0, 0]  # solid phase composed only by API (C)
        solid_cry = SolidPhase(path_phys, x_distrib=x_gr, distrib=distrib_init, mass_frac=massfrac_solid)
        flst.CR01 = MSMPR(target_comp='C', method='1D-FVM', scale=1e-9,)
        flst.CR01.Kinetics = cryst_kin
        flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=278.15)
        flst.CR01.Phases = (liquid_init_cry, solid_cry,)
        time_CR01 = time_R01
        flst.HOLD01 = DynamicCollector()
        # Filter definition
        alpha = 1e11
        Rm = 1e10
        deltaP = 101325 # In Pa
        filt_area = 200  # cm**2
        diam = np.sqrt(4/np.pi * filt_area) / 100  # m
        flst.F01 = Filter(diam, alpha, Rm)
        runargs_R01 = {'runtime': time_R01}
        sundials = {'maxh': 60} 
        runargs_CR01 = {'runtime': time_CR01, 'sundials_opts': sundials} # Runtime
        runargs_hold = {'runtime': time_CR01,'time_grid': np.arange(10800, 3600 * 10, 100)} 
        runargs_F01 = {'runtime': None, 'deltaP': deltaP} 
        run_kwargs = {'R01': runargs_R01,'CR01': runargs_CR01,'HOLD01': runargs_hold,'F01': runargs_F01}
        run_kwargs = make_non_verbose(run_kwargs)
        flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)
        print('PFR_MSMPR_HOLD_FILT flowsheet simulation complete')


class TestExtraInlets(unittest.TestCase):
    """One upstream unit plus fresh feeds into a single refactored vessel.

    The flowsheet hands upstream material over by assigning Inlet, which
    replaces the vessel's earlier Inlet. Feeds given through extra_inlets
    must survive that handover and be fed alongside the upstream stream.

    Runs on the refactored units with the scipy backend, so it needs no
    Assimulo. R02 is semibatch: nothing leaves it, so everything that enters
    must be found in its holdup, which makes the balances exact up to solver
    and quadrature error.
    """

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'compound_database.json')
    RXNS = ['A + B --> C', 'C + A --> D']
    TEMP = 313.15
    RUNTIME = 3600.0
    IDX_SOLVENT = 4
    # A + B --> C and C + A --> D: every B consumed makes one C, every C
    # consumed makes one D, so moles of B + C + D are conserved.
    IDX_BCD = [1, 2, 3]

    # Solver and trapezoid error measured at about 2e-7 relative; this
    # bound catches a dropped feed (a third of R02's intake) with room
    # to spare.
    RTOL = 1e-5

    def _feed(self, mole_conc, vol_flow):
        return NewLiquidStream(self.PATH, temp=self.TEMP,
                               mole_conc=np.asarray(mole_conc, dtype=float),
                               vol_flow=vol_flow, name_solv='solvent')

    def _reactor(self, cls, controller, inlet=None):
        unit = cls(integrator=ScipyBackend(options={'maxh': 60}),
                   h_conv=10000.0, diam=0.4, controller=controller)
        unit.Phases = NewLiquidPhase(
            self.PATH, temp=self.TEMP,
            mole_conc=np.array([0.33, 0.33, 0, 0, 0]),
            vol=0.06, name_solv='solvent')
        unit.RxnKinetics = RxnKinetics(
            path=self.PATH, rxn_list=self.RXNS,
            k_params=np.array([2.654e4, 5.3e2]),
            ea_params=np.array([4.0e4, 3.0e4]))

        if inlet is not None:
            unit.Inlet = inlet

        return unit

    def _upstream_feed(self):
        return self._feed([0.5, 0.5, 0, 0, 0], 1e-5)

    def _run(self, extra_inlets):
        """CSTR R01 --> semibatch R02, R02 given a placeholder Inlet."""
        flst = SimulationExec(self.PATH, flowsheet='R01 --> R02')

        flst.R01 = self._reactor(
            NewContReactor,
            ContinuousVesselController(temp_func=lambda t: self.TEMP),
            inlet=self._upstream_feed())

        # The placeholder the flowsheet is meant to replace.
        flst.R02 = self._reactor(
            NewSemiReactor,
            SimpleTemperatureController(temp_func=lambda t: self.TEMP),
            inlet=self._feed([0.5, 0.5, 0, 0, 0], 1e-5))

        if extra_inlets is not None:
            flst.R02.extra_inlets = extra_inlets

        run = {name: {'runtime': self.RUNTIME, 'verbose': False}
               for name in ('R01', 'R02')}
        flst.SolveFlowsheet(kwargs_run=run, verbose=False)

        return flst

    def _mw(self, flst):
        return np.asarray(flst.R01.Phases.Liquids[0].mw, dtype=float)

    def _upstream_outflow(self, flst):
        """Mass of each species R01 discharged, from its own trajectory."""
        result = flst.R01.result
        time = np.asarray(result.time)
        mole_conc = np.asarray(result.mole_conc_liquid0)    # kmol/m3
        vol_flow = np.asarray(result.outlet_vol_flow)       # m3/s

        mass_flow_j = mole_conc * self._mw(flst) * vol_flow[:, np.newaxis]

        return np.array([trapezoidal_rule(time, mass_flow_j[:, j])
                         for j in range(mass_flow_j.shape[1])])

    def _feed_mass(self, streams):
        """Mass of each species a constant feed delivers over the run."""
        return sum(np.asarray(s.mass_flow) * np.asarray(s.mass_frac)
                   for s in streams) * self.RUNTIME

    def _accumulated(self, unit):
        mass_j = np.asarray(unit.result.mass_j_liquid0)
        return mass_j[-1] - mass_j[0]

    def _assert_closes(self, label, accumulated, entered, throughput=None):
        """Residual relative to throughput, which defaults to `entered`.

        A vessel near steady state accumulates a small difference of large
        flows, so dividing by the accumulation would inflate an error that
        is tiny next to what actually passed through.
        """
        if throughput is None:
            throughput = entered
        scale = max(abs(throughput), 1e-30)
        rel = abs(accumulated - entered) / scale
        self.assertLess(
            rel, self.RTOL,
            '%s does not close: accumulated %.6g, entered %.6g (rel %.2e)'
            % (label, accumulated, entered, rel))

    def test_upstream_plus_extra_inlets_mass_balance(self):
        solvent = self._feed([0, 0, 0, 0, 0], 2e-6)
        b_feed = self._feed([0, 0.8, 0, 0, 0], 3e-6)
        extras = [solvent, b_feed]

        flst = self._run(extras)

        # The upstream stream replaced the placeholder; the extras survived.
        connections = flst.R02.inlet_connections
        self.assertEqual(len(connections), 3)
        self.assertIs(connections[1].stream.Phases[0], solvent)
        self.assertIs(connections[2].stream.Phases[0], b_feed)
        self.assertIsNotNone(
            getattr(connections[0].stream, 'y_upstream', None)
            or connections[0].stream.Phases[0].y_upstream,
            'connection 0 should carry the upstream trajectory')

        upstream_out = self._upstream_outflow(flst)
        mw = self._mw(flst)

        # Control: R01 must balance on its own, or the outflow integral
        # used below is not a trustworthy measure of what left it.
        r01_in = self._feed_mass([self._upstream_feed()])
        r01_acc = self._accumulated(flst.R01)
        self._assert_closes('R01 total mass', r01_acc.sum(),
                            r01_in.sum() - upstream_out.sum(),
                            throughput=r01_in.sum())

        extra_in = self._feed_mass(extras)
        r02_acc = self._accumulated(flst.R02)

        # Guard against a vacuous pass: the extras must be a real share of
        # what R02 took in, so dropping them could not hide in the tolerance.
        self.assertGreater(extra_in.sum(), 0.1 * r02_acc.sum())

        self._assert_closes('R02 total mass', r02_acc.sum(),
                            upstream_out.sum() + extra_in.sum())

        i = self.IDX_SOLVENT
        self._assert_closes('R02 solvent', r02_acc[i],
                            upstream_out[i] + extra_in[i])

        bcd = self.IDX_BCD
        self._assert_closes(
            'R02 moles of B + C + D',
            (r02_acc[bcd] / mw[bcd]).sum(),
            ((upstream_out[bcd] + extra_in[bcd]) / mw[bcd]).sum())

    def test_placeholder_inlet_replaced_without_extras(self):
        """Without extra_inlets the handover replaces, as it always has."""
        flst = self._run(extra_inlets=None)

        self.assertEqual(len(flst.R02.inlet_connections), 1)

        # Everything R02 gained came from R01: the placeholder feed is gone.
        upstream_out = self._upstream_outflow(flst)
        self._assert_closes('R02 total mass',
                            self._accumulated(flst.R02).sum(),
                            upstream_out.sum())

    def test_extra_inlets_survive_reassigning_inlet(self):
        unit = self._reactor(
            NewSemiReactor,
            SimpleTemperatureController(temp_func=lambda t: self.TEMP))
        extra = self._feed([0, 0, 0, 0, 0], 2e-6)
        unit.extra_inlets = extra

        first, second = self._upstream_feed(), self._upstream_feed()
        unit.Inlet = first
        unit.Inlet = second

        streams = [c.stream.Phases[0] for c in unit.inlet_connections]
        self.assertEqual(len(streams), 2)
        self.assertIs(streams[0], second)
        self.assertIs(streams[1], extra)

        # Connections set directly are kept too, with the extras after them.
        direct = unit.inlet_connections[:1]
        unit.inlet_connections = direct
        self.assertEqual(len(unit.inlet_connections), 2)
        self.assertIs(unit.inlet_connections[0], direct[0])
        self.assertIs(unit.inlet_connections[1].stream.Phases[0], extra)

        unit.extra_inlets = []
        self.assertEqual(len(unit.inlet_connections), 1)
        self.assertIs(unit.inlet_connections[0].stream.Phases[0], second)


if __name__ == '__main__':
    unittest.main()
    