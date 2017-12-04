typedef struct HCIManager {
    /* private: */
    char * prefix;
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
    NO_ACTION = 0,
    STOP = 1,
    TIMEOUT = 2,
    AUTO_CHECKPOINT = 3,
    CHECKPOINT = 4,
    TERMINATE = 5,
    IOCTL = 6,
};

typedef struct HCIAction
{
    enum HCIActionType type;
    int write_snapshot;
    int write_fof;
} HCIAction;

extern HCIManager HCI_DEFAULT_MANAGER[];

void
hci_init(HCIManager * manager, char * prefix, double TimeLimitCPU, double AutoCheckPointTime);

int
hci_query(HCIManager * manager, HCIAction * action);

int
hci_override_now(HCIManager * manager, double now);
