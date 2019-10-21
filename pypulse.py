import re
import sys
import numpy as np
try:
	import quaternion as quat
except ImportError:
	print("Oops, you don't have the quaternion extension to numpy.")
	print("Download it here: https://github.com/moble/quaternion")

PI = np.pi

plt = None
def get_matplotlib():
	global plt
	if not plt:
		try:
			from matplotlib import pyplot as plt
			return plt
		except ImportError:
			print("Oops, you don't have the matplotlib module")
			print("This is requred to make plots")
			return
	else:
		return plt


class Spins(object):

	def __init__(self, number=200, initial_magnetisation=(0.,0.,1.), 
		bandwidth=6000.0, offset=0.0, spectrometer_freq=600.0, 
		r1=1.0, r2=2.0, ieq=1.0):
		"""
		Spins object that can be used to track pulse trajectories with pulses

		Parameters
		----------
		number : int
			the number of linearly spaced spin offset frequencies
			default is 200 points
		initial_magnetisation : array of floats
			the initial magnetisation vector in cartesian coordinates [x,y,z]
			defauls is aligned with the z-axis [0.0, 0.0, 1.0]
		bandwidth : float
			the bandwidth of the spin offset frequencies in Hz
			default is 6000.0 Hz
		offset : float
			the offset of the spin frequencies and centre of the spectrum in Hz
			default is 0.0 Hz
		spectrometer_freq : float
			the spectrometer frequency in MHz
		r1 : float
			the R1 = 1/T1 longitudinal relaxation rate in Hz
			defualt is 1.0 Hz
		r2 : float
			the R2 = 1/T2 transverse relaxation rate in Hz
			defualt is 2.0 Hz
		ieq : float
			the equilibrium magnetisation amplitude along the z-axis.
			as T -> infinity, the magnetisation will relax exponentially towards
			ieq, as characterised by <r1>
			defualt is 1.0 Hz
		"""
		self.initial_magnetisation = np.array(initial_magnetisation, dtype=float)
		self.spectrometer_freq = float(spectrometer_freq)
		self.r1 = float(r1)
		self.r2 = float(r2)
		self.ieq = float(ieq)

		self.offsets = 2 * PI * (offset + bandwidth * np.linspace(-0.5, 0.5, number))
		self.q_offsets = quat.as_quat_array([[0.,0.,0.,i] for i in self.offsets])

		self.reset()

	def __len__(self):
		return len(self.offsets)

	def get_hz_scale(self):
		return self.offsets / (2*PI)

	def get_ppm_scale(self):
		return self.get_hz_scale() / self.spectrometer_freq

	def reset(self):
		"""
		Reset all spins to their initial vector specified by 'init_mag'

		"""
		tmp = np.tile([0]+list(self.initial_magnetisation), len(self)).reshape(len(self),4)
		self.magnetisation = quat.as_quat_array(tmp)

	def get_spins(self):
		"""
		Get the spin magnetisation as an array of 3-vectors

		Returns
		-------
		array : numpy float array
			array of magnetisation vectors (x,y,z) 
		"""
		return quat.as_float_array(self.magnetisation).T[1:].T

	def get_spin_at_ppm(self, ppm_value):
		i = np.argmin(np.abs(self.get_ppm_scale()-ppm_value))
		return self.magnetisation[i]

	def plot(self, scale='Hz', figure=None):
		"""
		Get a plot object of the current spin magnetisation

		Arguments
		---------
		scale : str
			either 'Hz' or 'ppm' for x-axis scale
			Note that <spectrometer_freq> attribute must be set for 'ppm' scale
			default scale is 'hz'
		figure : matplotlib figure object, optional
			when None (default) autonomously calls 'show' on generated plot
			when set to matplotlib figure object, plots on this figure and returns it

		Returns
		-------
		figure : matplotlib object
			call the 'show' function on returned figure to view plot
		"""
		plt = get_matplotlib()
		if not figure:
			fig = plt.figure()
		else:
			fig = figure
		ax = fig.add_subplot(111)
		x, y, z = self.get_spins().T
		if scale == 'Hz':
			s = self.get_hz_scale()
			invert = False
		elif scale == 'ppm':
			s = self.get_ppm_scale()
			invert = True
		else:
			raise ValueError("That scale type is not supported. Must be 'Hz' or 'ppm'")
		ax.plot(s, x, label='x', c='r')
		ax.plot(s, y, label='y', c='g')
		ax.plot(s, z, label='z', c='b')
		ax.set_ylim(-1.1, 1.1)
		ax.legend()
		if invert:
			ax.invert_xaxis()
		ax.set_xlabel("offset /{}".format(scale))
		if not figure:
			plt.show(fig)
			return fig
		else:
			return fig


class ShapedPulse(object):

	def __init__(self, pulse_length=None, normalised_shape=None, 
		amplitudes=None, name=None, parameters={}):
		"""
		Shaped pulse class that is used to store pulse shapes and calculate
		spin evolutions.

		Note that only <amliptudes> and <pulse_length> must be set to make pulses.

		Parameters
		----------
		pulse_length : float
			the length of the pulse in seconds
		normalised_shape : numpy array of floats
			the shaped pulse normalised with maximum amplitude 1.0.
			Only x and y shape amplitudes are stored
		amplitudes : numpy array of quaternions
			numpy quaternion array of (w,x,y,z) vectors describing the pulse B1
			values in Tesla. Note w=z=0.0 usually
		name : string
			a label for the pulse to keep track of pulse history.
		parameters : dict
			additional parameters used for calibrating shaped pulses.
			This may contain band-width factors and integration factors etc.
		"""
		self.pulse_length = pulse_length
		self.normalised_shape = normalised_shape
		self.amplitudes = amplitudes
		self.name = name
		self.parameters = parameters

	def __str__(self):
		if self.name is None:
			name = ""
		else:
			name = self.name
		if self.pulse_length is None:
			pl = ""
		else:
			pl = "{:3.2f} us".format(self.pulse_length * 1E6)
		return "<ShapedPulse: {0}, {1}>".format(name, pl)

	def __len__(self):
		if self.amplitudes is not None:
			return len(self.amplitudes)
		elif self.normalised_shape is not None:
			return len(self.normalised_shape)
		else:
			raise AttributeError("You must set a shape before obtaining its length")

	@property
	def increment(self):
		return self.pulse_length / len(self)

	@increment.setter
	def increment(self, value):
		self.pulse_length = value * len(self)

	def calibrate(self, maximum_amplitude, pulse_length=None, offset=0.0):
		"""
		Calibrate pulse amplitudes 

		Parameters
		----------
		maximum_amplitude : float
			the maximum amplitude in 
		pulse_length : float
			the length of the pulse
		offset : float
			the pulse offset in Hz

		"""
		if pulse_length is not None:
			self.pulse_length = pulse_length
		n = len(self.normalised_shape)
		tmp = np.vstack([np.zeros(n), 
			maximum_amplitude * self.normalised_shape.T, 
			-2*PI*offset*np.ones(n)]).T
		self.amplitudes = quat.as_quat_array(tmp)
		
	def calibrate_from_bandwidth(self, bandwidth, offset=0.0):
		"""
		Calibrate shaped pulse amplitudes for a given bandwidth.
		Note this method requires the following parameters to already
		be set within the parameters dictionary:
			TOTROT : float
				total rotation at zero offset for the pulse
			BWFAC : float
				the bandwidth factor
			INTEGFAC : float
				the integration factor
		If using the 'import_bruker_shape' function, these are automatically set

		Arguments
		---------
		bandwidth : float
			the bandwidth in Hz of the desired calibrated pulse

		"""
		rotation = self.parameters.get('TOTROT') * (PI/180.0)
		bw_fac = self.parameters.get('BWFAC')
		int_fac = self.parameters.get('INTEGFAC')

		pulse_length = bw_fac / bandwidth
		maximum_amplitude = (rotation / pulse_length) / int_fac
		self.calibrate(maximum_amplitude, pulse_length=pulse_length, offset=offset)

	def pulse(self, spins):
		"""
		Calculate pulse evolution as applied to spin object

		Arguments
		---------
		spins : <Spins> object

		Returns
		-------
		array: quaternion array
			evolved spins are returned as a quaternion array and are also 
			stored in the 'mag' attribute of the 'Spins' object argument

		"""
		if hasattr(self, 'amplitudes'):
			q = quat.as_quat_array([1.,0.,0.,0.] * len(spins))
			incr = self.increment
			for rf in self.amplitudes:
				beff = (spins.q_offsets + rf) * incr
				q = np.exp(-0.5 * beff) * q

			mag = q * spins.magnetisation * np.conjugate(q)
			spins.magnetisation = mag
			return mag
		else:
			print("Cannot pulse spins. Pulse calibration may be required first:")
			print("Call the <calibrate> method to set pulse amplitudes.")


	def pulse_relaxation(self, spins):
		"""
		Calculate pulse evolution as applied to spin object including relaxation

		Arguments
		---------
		spins : <Spins> object

		Returns
		-------
		array: quaternion array
			evolved spins are returned as a quaternion array and are also 
			stored in the 'mag' attribute of the 'Spins' object argument

		"""
		if hasattr(self, 'amplitudes'):
			incr = self.increment
			tran_evol = lambda x: x * (1. - spins.r2 * incr)
			long_evol = lambda x: x - (spins.r1 * incr * (x - spins.ieq))
			mag = spins.magnetisation
			for rf in self.amplitudes:
				beff = (spins.q_offsets + rf) * incr
				q = np.exp(0.5 * beff)
				mag = q * mag * np.conjugate(q)
				mag = quat.as_float_array(mag).T
				mag[1] = tran_evol(mag[1])
				mag[2] = tran_evol(mag[2])
				mag[3] = long_evol(mag[3])
				mag = quat.as_quat_array(mag.T)
			return mag
		else:
			print("Cannot pulse spins. Pulse calibration is required first:")
			print("Call the 'calibrate_soft' or 'calibrate_hard' method")


	def plot_amplitude(self, figure=None):
		plt = get_matplotlib()
		if not figure:
			fig = plt.figure()
		else:
			fig = figure
		x, y = self.normalised_shape.T
		t = np.arange(len(self))*self.pulse_length
		ax = fig.add_subplot(111)
		ax.plot(t, x, label='x norm. amplitude')
		ax.plot(t, y, label='y norm. amplitude')
		ax.set_xlabel("time /s")
		ax.set_ylabel("Amplitude /rad/s")
		ax.legend()
		if not figure:
			plt.show(fig)
			return fig
		else:
			return fig

	def plot_phase(self, figure=None):
		plt = get_matplotlib()
		if not figure:
			fig = plt.figure()
		else:
			fig = figure
		x, y = self.normalised_shape.T
		c = np.zeros(len(self), dtype=complex)
		c.real = x
		c.imag = y
		amp = np.abs(c)
		phase = np.angle(c)
		t = np.arange(len(self))*self.pulse_length
		ax1 = fig.add_subplot(111)
		ax1.plot(t, amp, 'r')
		ax2 = ax1.twinx()
		ax2.plot(t, phase/(2*np.pi), 'b')
		ax1.tick_params(axis='y', labelcolor='r')
		ax2.tick_params(axis='y', labelcolor='b')
		ax1.set_xlabel("time /s")
		ax1.set_ylabel('normalised amplitude', color='r')
		ax2.set_ylabel('phase /rad', color='b')
		if not figure:
			plt.show(fig)
			return fig
		else:
			return fig

	
def import_bruker_shape(file_name):

	def get_par(var, dic, typ):
		if dic==None:
			try:
				return typ(var)
			except ValueError as err:
				print("Parameter '{}' has non-standard type".format(var))
				print("It has been set to 'None' which may affect future calculations")
				return None
		else:
			try:
				return typ(dic[var])
			except (KeyError, ValueError) as err:
				print("Parameter '{}' has non-standard type or cannot be found".format(var))
				print("It has been set to 'None' which may affect future calculations")
				return None

	info = {}
	shape = []

	with open(file_name, 'r') as o:
		info_re = re.compile('##.*')
		param_re = re.compile('##\$.*')
		shape_re = re.compile('.*,.*')

		for line in o:
			if '##END' in line: break

			elif param_re.match(line):
				name = line.split('=')[0][3:]
				value = "".join(line.split('=')[1:])[:-1]
				info[name] = value

			elif info_re.match(line):
				name = line.split('=')[0][2:]
				value = "".join(line.split('=')[1:])[:-1]
				info[name] = value

			elif shape_re.match(line):
				try:
					val = map(float, line.split(','))
					shape.append(val)
				except ValueError:
					print("The shape pulse values contain non-standard float values")
					raise

			else: print("Line ignored: \n{}".format(line))
	points = len(shape)
	maxx = get_par('MAXX', info, float)
	name = get_par('TITLE', info, str).lstrip()

	amps, phases = zip(*shape)
	if maxx:
		amps = np.array(amps)/maxx
	else:
		amps = np.array(amps)

	phases = (np.array(phases)-180.0)*(PI/180.)
	zeros = np.zeros(len(amps))
	tmp = (amps * np.exp(1j*phases)).round(7)
	norm_shape = np.array([tmp.real, tmp.imag]).T

	params = {
		'TOTROT': get_par('SHAPE_TOTROT', info, float),
		'BWFAC': get_par('SHAPE_BWFAC', info, float),
		'INTEGFAC': get_par('SHAPE_INTEGFAC', info, float)
	}

	s = ShapedPulse(normalised_shape=norm_shape, name=name, parameters=params)
	return s


def make_delay(length, name='free_precession'):
	normalised_shape = np.array([[0.,0.]])
	s = ShapedPulse(normalised_shape=normalised_shape, name=name)
	s.calibrate(maximum_amplitude=0.0, pulse_length=length)
	return s


def make_hard_pulse(pulse_length=8.0E-6, rotation=PI/4, phase=0.0, offset=0.0):
	"""
	Make a hard pulse

	Parameters
	----------
	pulse_length : float
		the length of the pulse in seconds
	rotation : float
		the flip angle in radians
	phase : float
		the pulse phase in radians
	offset : float
		the pulse offset in Hz
	"""
	c = np.exp(1j*phase)
	normalised_shape = np.array([(c.real, c.imag)])
	shape = ShapedPulse(normalised_shape=normalised_shape, name='hard_pulse')
	amplitude = rotation / pulse_length
	shape.calibrate(amplitude, pulse_length=pulse_length, offset=offset)
	return shape



