
class SubElement:
	def __init__(self, coords_):
		
		self.r0 = coords_[0] ## center
		self.r1 = coords_[1] ## Sn - Cu6Sn5 bound
		self.r2 = coords_[2] ## Cu6Sn5 - Cu3Sn bound
		self.r3 = coords_[3] ## Cu3Sn - Cu bound
		self.r4 = coords_[4] ## Cu - Nb bound
		self.r5 = coords_[5] ## external radius
	
	def width(self):
		return self.r5 - self.r0


sx0 = 0    
sx1 = 15e-6
sx2 = 17e-6
sx3 = 20e-6 
sx4 = 25e-6 
sx5 = 30e-6
sx=[sx0,sx1,sx2,sx3,sx4, sx5]
a1=SubElement(sx)
print(a1.width())