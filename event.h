#ifndef _EVENT_H_
#define _EVENT_H_


typedef struct EventHandler EventHandler;

typedef struct EventSpec EventSpec;
typedef struct {
    int unused;
} EIBase;

typedef int (*eventfunc) (EIBase * event, void * userdata);

struct EventHandler
{
    eventfunc func;
    void * userdata;
};

#define MAXEH 8

struct EventSpec
{
    char name[32];
    int used;
    EventHandler h[MAXEH];
};

int
event_listen(EventSpec * e, eventfunc func, void * userdata);

int
event_unlisten(EventSpec * e, eventfunc func, void * userdata);

int
event_emit(EventSpec * eh, EIBase * event);

/* A new particle is formed by spliting an existing particle. */
extern EventSpec EventSlotsFork;
/* GC is done, things may have been violated. */
extern EventSpec EventSlotsAfterGC;

#endif
