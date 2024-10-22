package org.datahub.domain.enums;

public enum TaskDispatchEnums {
    TAKE_ONCE(4);

    private final int value;

    TaskDispatchEnums(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
