#include "event.h"
#include <stdlib.h>
#include <string.h>

int
event_listen(EventSpec * eh, eventfunc func, void * userdata)
{
    int i;
    for(i = 0; i < eh->used; i ++) {
        if(eh->h[i].func == func && eh->h[i].userdata == userdata) {
            return 0;
        }
    }
    if(eh->used == MAXEH) {
        /* overflown */
        abort();
    }

    eh->h[eh->used].func = func;
    eh->h[eh->used].userdata = userdata;
    eh->used ++;
    return 0;
}

int
event_unlisten(EventSpec * eh, eventfunc func, void * userdata)
{
    int i;
    for(i = 0; i < eh->used; i ++) {
        if(eh->h[i].func == func && eh->h[i].userdata == userdata) {
            break;
        }
    }
    if(i == eh->used || eh->used == 0) {
        return 1;
    }
    memmove(&eh->h[i], &eh->h[i+1], sizeof(eh->h[0]) * (eh->used - i - 1));
    eh->used --;
    return 0;
}


int
event_emit(EventSpec * eh, EIBase * event)
{

    int i;
    for(i = 0; i < eh->used; i ++) {
        eh->h[i].func(event, eh->h[i].userdata);
    }
    return 0;
}

EventSpec EventSlotsFork = {"SlotsFork", 0, {{0}}};
EventSpec EventSlotsAfterGC = {"SlotsAfterGC", 0, {{0}}};


