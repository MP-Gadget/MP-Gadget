/*Tests for the drift factor module.*/
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <stdint.h>

#include "stub.h"
#include "libgadget/utils/endrun.h"
#include "libgadget/metal_return.h"
#include "libgadget/slotsmanager.h"
#include "libgadget/metal_tables.h"

void test_yields(void ** state)
{
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    set_metal_params(1.3e-3);

    struct interps interp;
    setup_metal_table_interp(&interp);
    /* Compute factor to normalise the total mass in the IMF to unity.*/
    double imf_norm = compute_imf_norm(gsl_work);
    assert_true(fabs(imf_norm - 0.624632) <  0.01);

    double agbyield = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 1, 40, gsl_work);
    double agbyield2 = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 1, SNAGBSWITCH, gsl_work);
    assert_true(fabs(agbyield / agbyield2 - 1) < 1e-3);
    /* Lifetime is about 200 Myr*/
    double agbyield3 = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 5, 40, gsl_work);

    /* Integrate the region of the IMF which contains SNII and AGB stars. The yields should never be larger than this*/
    gsl_function ff = {chabrier_mass, NULL};
    double agbmax, sniimax, abserr;
    gsl_integration_qag(&ff, agb_total_mass[0], SNAGBSWITCH, 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &agbmax, &abserr);
    gsl_integration_qag(&ff, SNAGBSWITCH, snii_masses[SNII_NMASS-1], 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &sniimax, &abserr);

    double sniiyield = compute_snii_yield(interp.snii_mass_interp, snii_total_mass, 0.01, 1, 40, gsl_work);

    double sn1a = sn1a_number(0, 1500, 0.679)*sn1a_total_metals;
    assert_true(sn1a < 1.3e-3);

    message(0, "agbyield %g max %g (in 200 Myr: %g)\n", agbyield, agbmax, agbyield3);
    message(0, "sniiyield %g max %g sn1a %g\n", sniiyield, sniimax, sn1a);
    message(0, "Total fraction of mass returned %g\n", (sniiyield + sn1a + agbyield)/imf_norm);
    assert_true(agbyield < agbmax);
    assert_true(sniiyield < sniimax);
    assert_true((sniiyield + sn1a + agbyield)/imf_norm < 1.);

    double masslow1, masshigh1;
    double masslow2, masshigh2;
    double masslowsum, masshighsum;
    find_mass_bin_limits(&masslow1, &masshigh1, 0, 30, 0.02, interp.lifetime_interp);
    find_mass_bin_limits(&masslow2, &masshigh2, 30, 60, 0.02, interp.lifetime_interp);
    find_mass_bin_limits(&masslowsum, &masshighsum, 0, 60, 0.02, interp.lifetime_interp);
    message(0, "0 - 30: %g %g 30 - 60 %g %g 0 - 60 %g %g\n", masslow1, masshigh1, masslow2, masshigh2, masslowsum, masshighsum);
    assert_true(fabs(masslow1 - masshigh2) < 0.01);
    assert_true(fabs(masslowsum - masslow2) < 0.01);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_yields),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
