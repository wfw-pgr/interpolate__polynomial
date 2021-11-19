import math, os, sys
import numpy as np

# ========================================================= #
# ===  interpolate__quintic2                            === #
# ========================================================= #

def interpolate__quintic2( xp=None, xd=None, fx=None, fpx=None, fppx=None, \
                           order=5, nDerivative=2 ):

    # ------------------------------------------------- #
    # --- [1] size check                            --- #
    # ------------------------------------------------- #
    nCoef   = order       + 1
    nEqType = nDerivative + 1
    nPoint  = xd.shape[0]

    if ( nPoint * nEqType != nCoef ):
        print( "[interpolate__quintic2.py] information is not sufficient.... [ERROR]" )
        print( "[interpolate__quintic2.py] nPoint  ::  {0}".format( nPoint  ) )
        print( "[interpolate__quintic2.py] nEqType ::  {0}".format( nEqType ) )
        print( "[interpolate__quintic2.py] nCoef   ::  {0}".format( nCoef   ) )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [2] matrix making                         --- #
    # ------------------------------------------------- #
    Amat    =  np.zeros( (nCoef,nCoef) )
    lhs     =  np.concatenate( [fx,fpx,fppx] )

    for iDeriv in range( nEqType ):
        irow  = nPoint * iDeriv
        for ik in range( 0, nPoint ):
            for jk in range( iDeriv, nCoef ):
                deriv_coef       = math.factorial( jk ) / math.factorial( jk-iDeriv )
                Amat[irow+ik,jk] = deriv_coef * xd[ik]**( float(jk-iDeriv) )
            
    # ------------------------------------------------- #
    # --- [3] solve coefficient                     --- #
    # ------------------------------------------------- #
    coef   = np.dot( np.linalg.inv( Amat ), lhs )

    # ------------------------------------------------- #
    # --- [4] interpolation                         --- #
    # ------------------------------------------------- #
    ret    = np.zeros( (xp.shape[0],) )
    for ik in range( nCoef ):
        ret[:] = ret[:] + coef[ik]*xp[:]**( float(ik) )
    return( ret )


# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):

    def arbitral__func( xin, coef=None, order=5 ):
        if ( coef is None ):
            coef = np.random.uniform( size=order+1 )
            print( coef )
        ret = np.zeros( (xin.shape[0],) )
        for ik in range( order+1 ):
            ret[:] = ret[:] + coef[ik]*xin[:]**( float(ik) )
        return( ret )

    xp = np.linspace( 0.0, 2.4, 101 )
    yp = arbitral__func( xp )

    ipt = xp.shape[0] // 2
    
    xd   = np.array( [ xp[1], xp[-2] ] )
    fx   = np.array( [ yp[1], yp[-2] ] )
    fpx  = np.array( [ (yp[2]-yp[0])/(xp[2]-xp[0]), (yp[-1]-yp[-3])/(xp[-1]-xp[-3]) ] )
    fppx = np.array( [ ( yp[ 2] -2*yp[ 1] + yp[ 0] ) / ( 0.5*( xp[ 2] - xp[ 0] ) )**2, \
                       ( yp[-1] -2*yp[-2] + yp[-3] ) / ( 0.5*( xp[-1] - xp[-3] ) )**2 ] )

    ret = interpolate__quintic2( xp=xp, xd=xd, fx=fx, fpx=fpx, fppx=fppx )

    diff = ret - yp
    stack = np.concatenate( [ret[:,None],yp[:,None],diff[:,None],], axis=1 )
    print( stack )
    
    import nkUtilities.plot1D       as pl1
    import nkUtilities.load__config as lcf
    
    x_,y_                    = 0, 1
    pngFile                  = "png/out.png"
    config                   = lcf.load__config()
    config["plt_xAutoRange"] = True
    config["plt_yAutoRange"] = True
    config["plt_xRange"]     = [ -1.2, +1.2 ]
    config["plt_yRange"]     = [ -1.2, +1.2 ]

    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=xp, yAxis=ret, label="interpolated", linestyle="-" , linewidth=0.8 )
    fig.add__plot( xAxis=xp, yAxis=yp,  label="original"    , linestyle="--", linewidth=0.8 )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()


# ------------------------------------------------- #
# --- obsolete primitive ver.                   --- #
# ------------------------------------------------- #
# #  -- [1-1] 0th order                           --  #
# iDeriv = 0
# i_flr  = nCoef//(nDerivative+1) * iDeriv
# for ik in range( 0, nCoef//(nDerivative+1) ):
#     for jk in range( 0, nCoef ):
#         Amat[i_flr+ik,jk] = xd[ik]**( float(jk) )

# #  -- [1-2] 1st derivative order                --  #
# iDeriv = 1
# i_flr  = nCoef//(nDerivative+1) * iDeriv
# for ik in range( 0, nCoef//(nDerivative+1) ):
#     for jk in range( 1, nCoef ):
#         Amat[i_flr+ik,jk] = float(jk) * xd[ik]**( float(jk-1) )

# #  -- [1-3] 2nd derivative order                --  #
# iDeriv = 2
# i_flr  = nCoef//(nDerivative+1) * iDeriv
# for ik in range( 0, nCoef//(nDerivative+1) ):
#     for jk in range( 2, nCoef ):
#         Amat[i_flr+ik,jk] = float(jk) * float(jk-1) * xd[ik]**( float(jk-2) )

