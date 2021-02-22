#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "stub.h"

#include <libgadget/hci.h>

char prefix[1024] = "XXXXXXXX";

static void
touch(char * prefix, char * b)
{
    char * fn = fastpm_strdup_printf("%s/%s", prefix, b);
    FILE * fp = fopen(fn, "w");
    myfree(fn);
    fclose(fp);
}

static int
exists(char * prefix, char * b)
{
    char * fn = fastpm_strdup_printf("%s/%s", prefix, b);
    FILE * fp = fopen(fn, "r");
    myfree(fn);
    if(fp) {
        fclose(fp);
        return 1;
    }
    return 0;
}

HCIManager manager[1] = {
    {.OVERRIDE_NOW = 1, ._now = 0.0}};


static void
test_hci_no_action(void ** state)
{
    HCIAction action[1];

    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 0);

    hci_query(manager, action);
    assert_int_equal(action->type, HCI_NO_ACTION);
    assert_int_equal(action->write_snapshot, 0);

}

static void
test_hci_auto_checkpoint(void ** state)
{
    HCIAction action[1];

    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    hci_override_now(manager, 0.0);
    hci_query(manager, action);

    hci_override_now(manager, 1.0);
    hci_query(manager, action);

    assert_int_equal(action->type, HCI_AUTO_CHECKPOINT);
    assert_int_equal(action->write_snapshot, 1);
    assert_int_equal(action->write_fof, 1);
    assert_true(manager->LongestTimeBetweenQueries == 1.0);
}

static void
test_hci_auto_checkpoint2(void ** state)
{

    HCIAction action[1];
    hci_override_now(manager, 1.0);
    hci_init(manager, prefix, 10.0, 1.0, 0);

    hci_override_now(manager, 2.0);
    hci_query(manager, action);

    hci_override_now(manager, 4.0);
    hci_query(manager, action);

    assert_true(manager->LongestTimeBetweenQueries == 2.0);
    assert_int_equal(action->type, HCI_AUTO_CHECKPOINT);
    assert_int_equal(action->write_snapshot, 1);
    assert_int_equal(action->write_fof, 0);
}

static void
test_hci_timeout(void ** state)
{
    HCIAction action[1];
    hci_override_now(manager, 1.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    hci_override_now(manager, 5.0);
    hci_query(manager, action);

    assert_true(manager->LongestTimeBetweenQueries == 4.0);

    hci_override_now(manager, 8.5);
    hci_query(manager, action);
    assert_int_equal(action->type, HCI_TIMEOUT);
    assert_int_equal(action->write_snapshot, 1);
}

static void
test_hci_stop(void ** state)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "stop");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    assert_false(exists(prefix, "stop"));

    assert_int_equal(action->type, HCI_STOP);
    assert_int_equal(action->write_snapshot, 1);
}

static void
test_hci_checkpoint(void ** state)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "checkpoint");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    assert_false(exists(prefix, "checkpoint"));

    assert_int_equal(action->type, HCI_CHECKPOINT);
    assert_int_equal(action->write_snapshot, 1);
}

static void
test_hci_terminate(void ** state)
{
    HCIAction action[1];
    hci_override_now(manager, 0.0);
    hci_init(manager, prefix, 10.0, 1.0, 1);

    touch(prefix, "terminate");
    hci_override_now(manager, 4.0);
    hci_query(manager, action);
    assert_false(exists(prefix, "terminate"));

    assert_int_equal(action->type, HCI_TERMINATE);
    assert_int_equal(action->write_snapshot, 0);
}

static int setup(void ** state)
{
    char * ret = mkdtemp(prefix);
    message(0, "UsingPrefix : '%s'\n", prefix);
    return !ret;
}

static int teardown(void ** state)
{
    remove(prefix);
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_hci_no_action),
        cmocka_unit_test(test_hci_auto_checkpoint),
        cmocka_unit_test(test_hci_auto_checkpoint2),
        cmocka_unit_test(test_hci_timeout),
        cmocka_unit_test(test_hci_stop),
        cmocka_unit_test(test_hci_checkpoint),
        cmocka_unit_test(test_hci_terminate),
    };
    return cmocka_run_group_tests_mpi(tests, setup, teardown);
}
