import unittest
import torch
import numpy as np
from tinycio import Codec, Color
from tinycio.numerics import *

class TestCodec(unittest.TestCase):
	
	def setUp(self):
		pass

	def tearDown(self):
		pass

	tol_10 = 0.1
	tol_100 = 0.5
	tol_1k = 5.
	tol_10k = 20.

	print_all = False

	def check_diff_torch(self, v1, v2, tol):
		return torch.allclose(
			v1,
			v2,
			atol=tol)

	def test_codec(self):
		cols_10 = [
		Color(0.01, 0.05, 0.02),
		Color(10, 0.5, 0.2),
		Color(0.5, 10, 0.2),
		Color(0.5, 0.2, 10.),
		Color(10., 10., 10.)
		]

		cols_100 = [
		Color(100, 0.05, 0.02),
		Color(0.05, 100, 0.02),
		Color(0.05, 0.02, 100.),
		Color(100., 100., 100.)
		]

		cols_1k = [
		Color(1000, 0.05, 0.02),
		Color(0.05, 1000, 0.02),
		Color(0.05, 0.02, 1000.),
		Color(1000., 1000., 1000.)
		]
		cols_10k = [		
		Color(0.05, 10000, 0.02),
		Color(0.05, 0.02, 10000.),
		Color(10000., 10000., 10000.)
		]
		np.set_printoptions(suppress=True)
		for col in cols_10:
			col = col.image()
			encoded = Codec.logluv_encode(col)
			decoded = Codec.logluv_decode(encoded)
			compare = self.check_diff_torch(col, decoded, self.tol_10)
			if not compare or self.print_all:
				print("Codec ERROR: ", np.abs(Float3(decoded) - Float3(col)))
				print(Float3(col).round(decimals=3), "vs", Float3(decoded).round(decimals=3))
			self.assertTrue(compare)
		for col in cols_100 or self.print_all:
			col = col.image()
			encoded = Codec.logluv_encode(col)
			decoded = Codec.logluv_decode(encoded)
			compare = self.check_diff_torch(col, decoded, self.tol_100)
			if not compare:
				print("Codec ERROR: ", np.abs(Float3(decoded) - Float3(col)))
				print(Float3(col).round(decimals=3), "vs", Float3(decoded).round(decimals=3))
			self.assertTrue(compare)
		for col in cols_1k or self.print_all:
			col = col.image()
			encoded = Codec.logluv_encode(col)
			decoded = Codec.logluv_decode(encoded)
			compare = self.check_diff_torch(col, decoded, self.tol_1k)
			if not compare:
				print("Codec ERROR: ", np.abs(Float3(decoded) - Float3(col)))
				print(Float3(col).round(decimals=3), "vs", Float3(decoded).round(decimals=3))
			self.assertTrue(compare)
		for col in cols_10k or self.print_all:
			col = col.image()
			encoded = Codec.logluv_encode(col)
			decoded = Codec.logluv_decode(encoded)
			compare = self.check_diff_torch(col, decoded, self.tol_10k)
			if not compare:
				print("Codec ERROR: ", np.abs(Float3(decoded) - Float3(col)))
				print(Float3(col).round(decimals=3), "vs", Float3(decoded).round(decimals=3))
			self.assertTrue(compare)

if __name__ == '__main__':
	unittest.main()