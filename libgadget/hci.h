#ifndef _HCI_H
#define _HCI_H

typedef struct HCIManager {
    /* private: */
    char * prefix;
    int FOFEnabled;
    double TimeLastCheckPoint;
    double AutoCheckPointTime;
    double LongestTimeBetweenQueries;
    double WallClockTimeLimit;
    double timer_query_begin;
    double timer_begin;

    /* for debugging: */
    int OVERRIDE_NOW;
    double _now;
} HCIManager;

enum HCIActionType {
    HCI_NO_ACTION = 0,
    HCI_STOP = 1,
    HCI_TIMEOUT = 2,
    HCI_AUTO_CHECKPOINT = 3,
    HCI_CHECKPOINT = 4,
    HCI_TERMINATE = 5,
    HCI_IOCTL = 6,
};

typedef struct HCIAction
{
    enum HCIActionType type;
    int write_snapshot;
    int write_fof;
} HCIAction;

void
hci_init(HCIManager * manager, char * prefix, double TimeLimitCPU, double AutoCheckPointTime, int FOFEnabled);

void
hci_action_init(HCIAction * action);

int
hci_query(HCIManager * manager, HCIAction * action);

void
hci_override_now(HCIManager * manager, double now);

#endif
