import sbmltoodepy.modelclasses
from scipy.integrate import odeint
import numpy as np
import operator
import math

class SBMLmodel(sbmltoodepy.modelclasses.Model):

	def __init__(self):

		self.p = {} #Dictionary of model parameters
		self.p['k1'] = sbmltoodepy.modelclasses.Parameter(0.75, 'k1', True, metadata = sbmltoodepy.modelclasses.SBMLMetadata("k1"))
		self.p['k2'] = sbmltoodepy.modelclasses.Parameter(0.25, 'k2', True, metadata = sbmltoodepy.modelclasses.SBMLMetadata("k2"))

		self.c = {} #Dictionary of compartments
		self.c['compartment'] = sbmltoodepy.modelclasses.Compartment(0.5, 3, True, metadata = sbmltoodepy.modelclasses.SBMLMetadata("compartment"))

		self.s = {} #Dictionary of chemical species
		self.s['S1'] = sbmltoodepy.modelclasses.Species(1.5, 'Amount', self.c['compartment'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("S1"))
		self.s['S2'] = sbmltoodepy.modelclasses.Species(2.0, 'Amount', self.c['compartment'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("S2"))
		self.s['S3'] = sbmltoodepy.modelclasses.Species(1.5, 'Amount', self.c['compartment'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("S3"))
		self.s['S3']._modifiedBy = 1
		self.s['S4'] = sbmltoodepy.modelclasses.Species(4.0, 'Amount', self.c['compartment'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("S4"))
		self.s['S4']._modifiedBy = 2

		self.r = {} #Dictionary of reactions
		self.r['reaction1'] = reaction1(self)

		self.f = {} #Dictionary of function definitions
		self.time = 0

		self.AssignmentRules()



	def AssignmentRules(self):

		return

	def RateS3(self):

		return 0.5 * self.p['k1'].value

	def RateS4(self):

		return -0.5 * self.p['k2'].value

	def _SolveReactions(self, y, t):

		self.time = t
		self.s['S1'].amount, self.s['S2'].amount, self.s['S3'].amount, self.s['S4'].amount = y
		self.AssignmentRules()

		rateRuleVector = np.array([ 0, 0, self.RateS3(), self.RateS4()], dtype = np.float64)

		stoichiometricMatrix = np.array([[-1.],[ 1.],[ 0.],[ 0.]], dtype = np.float64)

		reactionVelocities = np.array([self.r['reaction1']()], dtype = np.float64)

		rateOfSpeciesChange = stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange

	def RunSimulation(self, deltaT, absoluteTolerance = 1e-12, relativeTolerance = 1e-6):

		finalTime = self.time + deltaT
		y0 = np.array([self.s['S1'].amount, self.s['S2'].amount, self.s['S3'].amount, self.s['S4'].amount], dtype = np.float64)
		self.s['S1'].amount, self.s['S2'].amount, self.s['S3'].amount, self.s['S4'].amount = odeint(self._SolveReactions, y0, [self.time, finalTime], atol = absoluteTolerance, rtol = relativeTolerance, mxstep=5000000)[-1]
		self.time = finalTime
		self.AssignmentRules()

class reaction1:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("reaction1")

	def __call__(self):
		return self.parent.c['compartment'].size * self.parent.p['k1'].value * self.parent.s['S1'].concentration

