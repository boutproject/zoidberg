import numpy as np

def GPI_view(nx=132,nz=128):
    
    bottom_left = [.4484,-5.9987,-.2948]
    bottom_right = [.4516,-6.0417,-.3106] 
    top_left = [.4503,-6.0246,-.2234] 
    top_right = [.4536,-6.0677,-.2392]
    
    left_edge_x  = np.linspace(bottom_left[0],top_left[0], nz)
    left_edge_z  = np.linspace(bottom_left[2],top_left[2], nz)
    left_edge_y  = np.linspace(bottom_left[1], top_left[1], nz)
    
    right_edge_x = np.linspace(bottom_right[0],top_right[0], nz)
    right_edge_z = np.linspace(bottom_right[2],top_right[2], nz)
    right_edge_y = np.linspace(bottom_right[1],top_right[1], nz)
    
    x_array = np.zeros((nx,nz))
    z_array = np.zeros((nx,nz))
    y_array = np.zeros((nx,nz))

    for z in np.arange(0,nz):
        line_x = np.linspace(left_edge_x[z],right_edge_x[z],nx)
        line_z = np.linspace(left_edge_z[z],right_edge_z[z],nx)
        line_y = np.linspace(left_edge_y[z],right_edge_y[z],nx)
        z_array[:,z] = line_z
        x_array[:,z] = line_x
        y_array[:,z] = line_y

    return x_array, y_array, z_array

def GPI_lpar(x,y,z, configuration=1, limit=5000):

    ### Use webservices to get connection length ###
    from osa import Client
    tracer = Client('http://esb:8280/services/FieldLineProxy?wsdl')
    points = tracer.types.Points3D()
    points.x1 = x.flatten()
    points.x2 = y.flatten()
    points.x3 = z.flatten()


    ### copied from webservices...#
    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = limit
    con.returnLoads = False
    task.connection = con

    # diff = tracer.types.LineDiffusion()
    # diff.diffusionCoeff = .00
    # diff.freePath = 0.1
    # diff.velocity = 5e4
    # task.diffusion = diff

    config = tracer.types.MagneticConfig()

    config.configIds = configuration  ## Just use machine IDs instead of coil currents because it's easier.

    ### This bit doesn't work when called as a function.  
    # config.coilsIds = range(160,230)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)


    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsIdsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    
    grid = tracer.types.Grid()
    grid.fieldSymmetry = 5

    cyl = tracer.types.CylindricalGrid()
    cyl.numR, cyl.numZ, cyl.numPhi = 181, 181, 481
    cyl.RMin, cyl.RMax, cyl.ZMin, cyl.ZMax = 4.05, 6.75, -1.35, 1.35
    grid.cylindrical = cyl

    machine = tracer.types.Machine(1)
    machine.grid.numX, machine.grid.numY, machine.grid.numZ = 500,500,100
    machine.grid.ZMin, machine.grid.ZMax = -1.5,1.5
    machine.grid.YMin, machine.grid.YMax = -7, 7
    machine.grid.XMin, machine.grid.XMax = -7, 7
    # machine.meshedModelsIds = [164]
    machine.assemblyIds = [12,14,8,9,13,21]

    config.grid = grid

    config.inverseField = False

    res_fwd = tracer.service.trace(points, config, task, machine)

    ###### end of copied code #######

    nx = x.shape[0]
    nz = x.shape[-1]
    
    lengths = np.zeros((nx*nz))
    
    for n in np.arange(0,nx*nz):
        lengths[n] = res_fwd.connection[n].length

    lengths = np.ndarray.reshape(lengths,(nx,nz))

    return lengths

def GPI_curvature(x,y,z):
    phi0 = 4.787
    
    from osa import Client
    from boututils import calculus as calc
    
    nx = x.shape[0]
    nz = x.shape[-1]
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    config = tracer.types.MagneticConfig()
    config.configIds = [1]


    #### Expand in phi to take derivatives
    phi_list = np.linspace(.99*phi0, 1.01*phi0, 9)
    r = x*np.cos(phi0) + y*np.sin(phi0)

    dz = z[0,1]-z[0,0]
    dx = x[1,0]-x[1,1]
    dphi = phi_list[1]-phi_list[0]

    r_extended = np.zeros((nx,9,nz))
    z_extended = np.zeros((nx,9,nz))
    phi_extended = np.zeros((nx,9,nz))
    
    
    for j in np.arange(0,phi_list.shape[0]):
        r_extended[:,j,:] = x*np.cos(phi_list[j]) + y*np.sin(phi_list[j])
        z_extended[:,j,:] = z
        phi_extended[:,j,:] = np.ones((nx,nz))*phi_list[j]
        
    points = tracer.types.Points3D()
    points.x1 = (r_extended*np.cos(phi_extended)).flatten()
    points.x2 = (r_extended*np.sin(phi_extended)).flatten()
    points.x3 = z_extended.flatten()

    res = tracer.service.magneticField(points, config)
    
    ## Reshape to 3d array
    Bx = np.ndarray.reshape(np.asarray(res.field.x1),(nx,9,nz))
    By = np.ndarray.reshape(np.asarray(res.field.x2),(nx,9,nz))
    
    bphi = -Bx*np.sin(phi_extended) + By*np.cos(phi_extended)
    Bz = np.ndarray.reshape(np.asarray(res.field.x3),(nx,9,nz))
    b2 = (Bx**2 + By**2 + Bz**2)
    Bx /= b2
    By /= b2
    Bz /= b2
    bphi /= b2

    dbphidz = np.zeros((nx,nz))
    dbrdz = np.zeros((nx,nz))
    dbzdphi = np.zeros((nx,nz))
    dbzdr = np.zeros((nx,nz))
    dbphidr = np.zeros((nx,nz))
    dbrdphi = np.zeros((nx,nz))
    curlbr = np.zeros((nx,nz))
    curlbphi = np.zeros((nx,nz))
    curlbz = np.zeros((nx,nz))
    
    for i in np.arange(0,nx):
        dbphidz[i,:] = calc.deriv(bphi[i,4,:])/dz
        dbrdz[i,:] = calc.deriv(Bx[i,4,:])/dz

    for k in np.arange(0,nz):
        dbzdr[:,k] = calc.deriv(Bz[:,4,k])/dx
        dbphidr[:,k] =  calc.deriv(bphi[:,4,k])/dx
        for i in np.arange(0,nx):
            dbzdphi[i,k] = calc.deriv(Bz[i,:,k])[4]/dphi
            dbrdphi[i,k] = calc.deriv(Bx[i,:,k])[4]/dphi
            curlbr[i,k]   = (dbzdphi[i,k] - dbphidz[i,k])  
            curlbphi[i,k] = (dbrdz[i,k] - dbzdr[i,k])
            curlbz[i,k]   = (dbphidr[i,k] - dbrdphi[i,k])


    return curlbr, curlbphi, curlbz

def GPI_Poincare(x,y,z, configuration=1):
    from osa import Client
    import matplotlib.pyplot as plt
    
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    config = tracer.types.MagneticConfig()
    config.configIds = configuration
    pos = tracer.types.Points3D()

    pos.x1 = np.linspace(0.95*np.mean(x[0,:]), 1.05*np.mean(x[-1,:]),80)
    pos.x2 = np.linspace(0.95*np.mean(y[0,:]), 1.05*np.mean(y[-1,:]),80)
    pos.x3 = np.linspace(0.95*np.mean(z[:,0]), 1.05*np.mean(z[:,-1]),80)
    
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints = 400
    poincare.phi0 = [4.787] ## This is where the poincare plane is (bean=0, triangle = pi/5.)
    
    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    
    res = tracer.service.trace(pos, config, task, None, None)
    
    for i in range(0, len(res.surfs)):
        plt.scatter(res.surfs[i].points.x1, res.surfs[i].points.x3, color="black", s=0.5)

    plt.xlim(x[0,0],x[-1,-1])
    plt.ylim(z[0,0],z[-1,-1])
    plt.show()    

    return res
