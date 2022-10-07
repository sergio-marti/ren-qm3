#include <stdio.h>
#include <stdlib.h>
#include <xtb.h>

void qm3_xtb_calc_( int *p_nQM, int *p_nMM, int *siz, double *dat ) {
    int       nQM = *p_nQM, nMM = *p_nMM; // fortran compliant...
    int       i, j, i3;
    int       *atn, uhf, *pc_num = NULL;
    double    chrg, ene, *xyz, *grd, *chg, *pc_chg = NULL, *pc_xyz = NULL, *pc_grd = NULL;

    double    x__Bohr = 1.0 / 0.52917721093;
    double    x__Ener = 2625.49963947555;

    // 3 + nQM [QM_chg] + 3 * nQM [QM_crd/grd] + nQM [QM_mul] + nMM [MM_chg] + 3 * nMM [MM_crd/grd]
    atn = (int*) malloc( nQM * sizeof( int ) );
    chg = (double*) malloc( nQM * sizeof( double ) );
    xyz = (double*) malloc( 3 * nQM * sizeof( double ) );
    grd = (double*) malloc( 3 * nQM * sizeof( double ) );
    for( i = 0; i < nQM; i++ ) {
        atn[i] = (int) dat[3+i];
        i3 = i * 3;
        j  = 3 + nQM + 3 * i;
        xyz[i3]   = dat[j]   * x__Bohr;
        xyz[i3+1] = dat[j+1] * x__Bohr;
        xyz[i3+2] = dat[j+2] * x__Bohr;
        grd[i3]   = 0.0;
        grd[i3+1] = 0.0;
        grd[i3+2] = 0.0;
    }

    xtb_TEnvironment env = xtb_newEnvironment();
    xtb_TCalculator calc = xtb_newCalculator();
    xtb_TResults res = xtb_newResults();

    chrg = dat[1];
    uhf = (int) dat[2];
    xtb_TMolecule mol = xtb_newMolecule( env, &nQM, atn, xyz, &chrg, &uhf, NULL, NULL);

    xtb_setVerbosity( env, XTB_VERBOSITY_MUTED );
//    xtb_setVerbosity( env, XTB_VERBOSITY_FULL );
    xtb_loadGFN2xTB( env, mol, calc, NULL );
//    xtb_setAccuracy( env, calc, 1.0 );
//    xtb_setElectronicTemp( env, calc, 300.0 );
    xtb_setMaxIter( env, calc, 100 );

    if( nMM > 0 ) {
        pc_num = (int*) malloc( nMM * sizeof( int ) );
        pc_chg = (double*) malloc( nMM * sizeof( double ) );
        pc_xyz = (double*) malloc( 3 * nMM * sizeof( double ) );
        pc_grd = (double*) malloc( 3 * nMM * sizeof( double ) );
        for( i = 0; i < nMM; i++ ) {
            // https://xtb-docs.readthedocs.io/en/latest/pcem.html
            // xtb%pcem%gam(ii) = xtb%xtbData%coulomb%chemicalHardness(numbers(ii))
            pc_num[i] = 7; // pcem_dummyatom = 7
            pc_chg[i] = dat[3+5*nQM+i];
            i3 = i * 3;
            j  = 3 + 5 * nQM + nMM + 3 * i;
            pc_xyz[i3]   = dat[j]   * x__Bohr;
            pc_xyz[i3+1] = dat[j+1] * x__Bohr;
            pc_xyz[i3+2] = dat[j+2] * x__Bohr;
            pc_grd[i3]   = 0.0;
            pc_grd[i3+1] = 0.0;
            pc_grd[i3+2] = 0.0;
        }
        xtb_setExternalCharges( env, calc, &nMM, pc_num, pc_chg, pc_xyz );
    }

    xtb_singlepoint( env, mol, calc, res );

    xtb_getEnergy( env, res, &ene );
    xtb_getCharges( env, res, chg );
    xtb_getGradient( env, res, grd );

    dat[0] = ene * x__Ener;
    for( i = 0; i < nQM; i++ ) {
        i3 = i * 3;
        j  = 3 + nQM + 3 * i;
        dat[j]   = grd[i3]   * x__Ener * x__Bohr;
        dat[j+1] = grd[i3+1] * x__Ener * x__Bohr;
        dat[j+2] = grd[i3+2] * x__Ener * x__Bohr;
        dat[3+4*nQM+i] = chg[i];
    }

    if( nMM > 0 ) {
        xtb_getPCGradient( env, res, pc_grd );
        for( i = 0; i < nMM; i++ ) {
            i3 = i * 3;
            j  = 3 + 5 * nQM + nMM + 3 * i;
            dat[j]   = pc_grd[i3]   *  x__Ener * x__Bohr;
            dat[j+1] = pc_grd[i3+1] *  x__Ener * x__Bohr;
            dat[j+2] = pc_grd[i3+2] *  x__Ener * x__Bohr;
        }
        xtb_releaseExternalCharges( env, calc );
    }

    xtb_delete( res );
    xtb_delete( calc );
    xtb_delete( mol );
    xtb_delete( env );

    free( atn ); free( chg ); free( xyz ); free( grd );
    free( pc_chg ); free( pc_xyz ); free( pc_grd ); free( pc_num );
}
