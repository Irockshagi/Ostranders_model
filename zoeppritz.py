import numpy as np
import matplotlib.pyplot as plt
import math



##########################################################
#
#       COMPUTATIONS HAPPEN BELOW HERE
#


#Log (model) calculation

dl1 = np.ones((top-dmin)/dd)
dl2 = np.ones(thickness/dd)
dl3 = np.ones((dmax-base)/dd)

#create Vp log in Depth domain
vpl1 = dl1*vp_mod[0]
vpl2 = dl2*vp_mod[1]
vpl3 = dl3*vp_mod[2]
vpl = np.concatenate((vpl1,vpl2,vpl3))

#create rho log in Depth domain
rhol1 = dl1*rho_mod[0]
rhol2 = dl2*rho_mod[1]
rhol3 = dl3*rho_mod[2]
rhol = np.concatenate((rhol1,rhol2,rhol3))

#create Vs  log in Depth domain
vsl1 = dl1*vs_mod[0]
vsl2 = dl2*vs_mod[1]
vsl3 = dl3*vs_mod[2]
vsl = np.concatenate((vsl1,vsl2,vsl3))

#create Poisson Ration log   in Depth domain
prl1=dl1*pr_mod[0]
prl2=dl2*pr_mod[1]
prl3=dl3*pr_mod[2]
prl = np.concatenate((prl1,prl2,prl3))
D = np.arange(dmin,dmax,dd)



#   Some handy constants
nlayers = len(vp_mod)
nint = nlayers - 1
nangles = int( (theta1_max-theta1_min)/theta1_step + 1)


#   Generate wavelet
if wvlt_type == 'ricker':
    wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
    
elif wvlt_type == 'bandpass':
    wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

#   Apply amplitude scale factor to wavelet (to match seismic amplitude values)
wvlt_amp = wvlt_scalar * wvlt_amp

#   Calculate reflectivities from model parameters
rc_zoep_pp = []
rc_zoep_ps = []
rc_zoep_tpp = []
rc_zoep_tps = []
theta1 = []
for i in range(0, nangles):
    theta1_buf = i*theta1_step + theta1_min
    rc_buf1 = rc_zoep(vp_mod[0], vs_mod[0], rho_mod[0], vp_mod[1], vs_mod[1], rho_mod[1], theta1_buf)
    rc_buf2 = rc_zoep(vp_mod[1], vs_mod[1], rho_mod[1], vp_mod[2], vs_mod[2], rho_mod[2], theta1_buf)
#    print rc_buf1

    
    theta1.append(theta1_buf)
    rc_zoep_pp.append([rc_buf1[0,0], rc_buf2[0,0]])
    rc_zoep_ps.append([rc_buf1[1,0], rc_buf2[1,0]])
    rc_zoep_tpp.append([rc_buf1[2,0], rc_buf2[1,0]])
    rc_zoep_tps.append([rc_buf1[3,0], rc_buf2[1,0]])

#   Define time sample vector for output model & traces
nsamp = int((tmax-tmin)/dt) + 1
t = []
for i in range(0,nsamp):
    t.append(i*dt)


syn_zoep_pp = []
lyr_times = []
print "\n\nStarting synthetic calcuations...\n"
for angle in range(0, nangles):
    
    dz_app = thickness
    
    #   To calculate apparent thickness of layer 2 based on incidence angle
    #   uncomment the following three rows (e.g. ray-synthetics)
    #p = ray_param(vp_mod[0], angle)
    #angle2 = math.degrees(math.asin(p*vp_mod[1]))
    #dz_app = thickness/math.cos(math.radians(angle2))
    
    #   Calculate interface depths
    z_int = [2000*2]
    z_int.append(z_int[0] + dz_app)
    
    #   Calculate interface times
    t_int = calc_times(z_int, vp_mod)
    lyr_times.append(t_int)
    
    #   Digitize 3-layer model
    rc = digitize_model(rc_zoep_pp[angle], t_int, t)

    #   Convolve wavelet with reflectivities
    syn_buf = np.convolve(rc, wvlt_amp, mode='same')
    syn_buf = list(syn_buf)
    syn_zoep_pp.append(syn_buf)
    print "Calculated angle %i" % (angle)


#    Convert data arrays from lists/tuples to numpy arrays    
syn_zoep_pp = np.array(syn_zoep_pp)
rc_zoep_pp = np.array(rc_zoep_pp)
t = np.array(t)


#   Calculate array indicies corresponding to top/base interfaces
lyr_times = np.array(lyr_times)
lyr_indx = np.array(np.round(lyr_times/dt), dtype='int16')
lyr1_indx = list(lyr_indx[:,0])
lyr2_indx = list(lyr_indx[:,1])


#   Copy convoved top/base reflectivity values to Lists for easier plotting
[ntrc, nsamp] = syn_zoep_pp.shape
line1 = []
line2 = []
for i in range(0, ntrc):
    line1.append(syn_zoep_pp[i,lyr1_indx[i]])
    line2.append(syn_zoep_pp[i,lyr2_indx[i]])



#   AVO inversion for NI and GRAD from analytic and convolved reflectivity
#   values and print the results to the command line.  Linear least squares
#   method is used for estimating NI and GRAD coefficients.
YzoepTop = np.array(rc_zoep_pp[:,0])
YzoepTop = YzoepTop.reshape((ntrc, 1))

YzoepBase = np.array(rc_zoep_pp[:,1])
YzoepBase = YzoepBase.reshape((ntrc, 1))

Yconv = np.array(line1)
Yconv = Yconv.reshape((ntrc, 1))

ones = np.ones(ntrc)
ones = ones.reshape((ntrc,1))

sintheta2 = np.sin(np.radians(np.arange(0, ntrc)))**2
sintheta2 = sintheta2.reshape((ntrc, 1))

X = np.hstack((ones, sintheta2))

#   ... matrix solution of normal equations
Azoep = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), YzoepTop)
AzoepBase = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), YzoepBase)
#Aconv = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Yconv)

print'\n\n'
print '  Method             NI         GRAD'
print '-------------------------------------'
print ' Top  Reflectifity%11.5f%12.5f' % (Azoep[0], Azoep[1])
print ' Base Reflectifity%11.5f%12.5f' % (AzoepBase[0], AzoepBase[1])
#print ' Convolved%10.5f%12.5f' % (Aconv[0], Aconv[1])


    
#   Create a "digital" time domain version of the input property model for 
#   easy plotting and comparison with the time synthetic traces
vp_dig = np.zeros(t.shape)
vs_dig = np.zeros(t.shape)
rho_dig = np.zeros(t.shape)

vp_dig[0:lyr1_indx[0]] = vp_mod[0]
vp_dig[(lyr1_indx[0]):lyr2_indx[0]] = vp_mod[1]
vp_dig[(lyr2_indx[0]):] = vp_mod[2]

vs_dig[0:lyr1_indx[0]] = vs_mod[0]
vs_dig[(lyr1_indx[0]):lyr2_indx[0]] = vs_mod[1]
vs_dig[(lyr2_indx[0]):] = vs_mod[2]

rho_dig[0:lyr1_indx[0]] = rho_mod[0]
rho_dig[(lyr1_indx[0]):lyr2_indx[0]] = rho_mod[1]
rho_dig[(lyr2_indx[0]):] = rho_mod[2]



