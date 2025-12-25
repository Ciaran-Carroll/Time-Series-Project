def check_dependencies():
    dependencies = {
        'yfinance': 'yfinance',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'statsmodels': 'statsmodels',
        'scikit-learn': 'sklearn'
    }

    print("Checking dependencies...\n")
    all_installed = True

    for name, import_name in dependencies.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15s} - Version {version}")
        except ImportError:
            print(f"✗ {name:15s} - NOT INSTALLED")
            all_installed = False

    if all_installed:
        print("\n✓ All dependencies installed successfully!")
    else:
        print("\n✗ Some dependencies missing. Run: pip install -r requirements.txt")

    return all_installed

if __name__ == "__main__":
    check_dependencies()
