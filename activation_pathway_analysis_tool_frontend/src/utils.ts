export const NodeColors: { [key: string]: string } = {
    'Conv2D': '#CBE4F9',
    'Dense': '#CDF5F6',
    'InputLayer': '#EFF9DA',
    'MaxPooling2D': '#FFFF00',
    'Flatten': '#F9EBDF',
    'Dropout': '#FF00FF',
    'Activation': '#FF8000',
    'GlobalAveragePooling2D': '#8000FF',
    'GlobalMaxPooling2D': '#0080FF',
    'BatchNormalization': '#D6CDEA',
    'Add': '#80FF00',
    'Concatenate': '#F9D8D6',
    'AveragePooling2D': '#800080',
    'ZeroPadding2D': '#808000',
    'UpSampling2D': '#008000',
    'Reshape': '#800000',
    'Permute': '#000080',
    'RepeatVector': '#808080',
    'Lambda': '#008080',
}

export function chunkify<T>(arr: T[], size: number): T[][] {
    return [...Array(Math.ceil(arr.length / size))].map((_, i) =>
        arr.slice(size * i, size + size * i)
    );
}