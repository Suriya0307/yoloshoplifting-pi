
try:
    import streamlit
    print("streamlit imported successfully")
except ImportError as e:
    print(f"streamlit import failed: {e}")

try:
    from detector import ShopliftingDetector
    print("detector imported successfully")
except ImportError as e:
    print(f"detector import failed: {e}")
except Exception as e:
    print(f"detector initialization failed: {e}")
