/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package ur_rna.RNAstructure.backend;

public class singlestructure {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected singlestructure(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(singlestructure obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        RNABackendJNI.delete_singlestructure(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public singlestructure(int sequencelength) {
    this(RNABackendJNI.new_singlestructure(sequencelength), true);
  }

  public void setEnergy(int value) {
    RNABackendJNI.singlestructure_energy_set(swigCPtr, this, value);
  }

  public int getEnergy() {
    return RNABackendJNI.singlestructure_energy_get(swigCPtr, this);
  }

  public void setCtlabel(String value) {
    RNABackendJNI.singlestructure_ctlabel_set(swigCPtr, this, value);
  }

  public String getCtlabel() {
    return RNABackendJNI.singlestructure_ctlabel_get(swigCPtr, this);
  }

  public int getSequenceLength() {
    return RNABackendJNI.singlestructure_getSequenceLength(swigCPtr, this);
  }

  public void setSequenceLength(int length) {
    RNABackendJNI.singlestructure_setSequenceLength(swigCPtr, this, length);
  }

  public int getBasePair(int basepos) {
    return RNABackendJNI.singlestructure_getBasePair(swigCPtr, this, basepos);
  }

  public void setBasePair(int basepos, int pairpos) {
    RNABackendJNI.singlestructure_setBasePair(swigCPtr, this, basepos, pairpos);
  }

}