const path = require('path');

const srcPath = (thePath) => path.resolve(__dirname, 'src', thePath);

module.exports = {
  webpack: {
    alias: {
      '@': srcPath(''),
      '@components': srcPath('components'),
      '@hooks': srcPath('app/hooks'),
      '@features': srcPath('features'),
      '@types': srcPath('types'),
      '@api': srcPath('api'),
      '@views': srcPath('components/LayerViews'),
      '@utils': srcPath('utils'),
      '@styles': srcPath('styles'),
    }
  }
};